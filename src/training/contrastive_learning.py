"""
对比学习训练模块 - 用于微调 Sentence-BERT

实现 SimCLR 风格的对比学习框架:
- InfoNCE 损失函数
- 批内负采样策略
- 支持 GPU 加速训练

Author: PathCL-EM Project
Date: 2025-01-10
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file as load_safetensors
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np

from augmentation import create_augmented_pair, batch_augment


def _load_cached_weights(model: SentenceTransformer, cache_model_path: Path) -> SentenceTransformer:
    """
    从缓存目录加载微调权重到已有模型，避免重新实例化 SentenceTransformer。

    只加载 model.safetensors 中的权重，跳过架构初始化和 tokenizer 加载，
    因此比 SentenceTransformer(path) 快很多。
    """
    weights_path = cache_model_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_safetensors(str(weights_path), device=str(model.device))
        model[0].auto_model.load_state_dict(state_dict)
    else:
        # 兜底：如果没有 safetensors 文件，回退到 pytorch_model.bin
        weights_path = cache_model_path / "pytorch_model.bin"
        state_dict = torch.load(str(weights_path), map_location=model.device)
        model[0].auto_model.load_state_dict(state_dict)
    return model


# ==================== 对比学习数据集 ====================

class ContrastiveDataset(Dataset):
    """对比学习数据集"""

    def __init__(self, entity_texts: List[str],
                 augmentation_methods1: List[str] = ['mask', 'shuffle'],
                 augmentation_methods2: List[str] = ['dropout', 'synonym']):
        """
        Args:
            entity_texts: 实体文本列表
            augmentation_methods1: 第一个视图的增强方法
            augmentation_methods2: 第二个视图的增强方法
        """
        self.entity_texts = entity_texts
        self.aug_methods1 = augmentation_methods1
        self.aug_methods2 = augmentation_methods2

    def __len__(self):
        return len(self.entity_texts)

    def __getitem__(self, idx):
        entity_text = self.entity_texts[idx]
        # 创建两个增强视图
        view1, view2 = create_augmented_pair(
            entity_text,
            method_set1=self.aug_methods1,
            method_set2=self.aug_methods2
        )
        return view1, view2


class SupervisedPairDataset(Dataset):
    """从标注数据加载正样本对"""

    def __init__(self, data_path: str, data_name: str):
        json_file = Path(data_path) / data_name / "labeled_pairs.json"
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        # 兼容新旧格式：新格式用 "pairs"（仅正样本），旧格式用 "triplets"（需过滤 label=1）
        if "pairs" in data:
            self.pairs = [(t[0], t[1]) for t in data["pairs"]]
        else:
            self.pairs = [(t[0], t[1]) for t in data["triplets"] if t[2] == 1]
        print(f"[SupervisedPairDataset] 加载 {len(self.pairs)} 个正样本对 (from {json_file})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # (text_a, text_b)


class ContrastiveLearner:
    """对比学习训练器"""

    def __init__(self,
                 base_model: SentenceTransformer,
                 temperature: float = 0.07,
                 device: str = 'cuda',
                 multi_gpu: bool = False):
        """
        Args:
            base_model: 预训练的 SentenceTransformer 模型
            temperature: InfoNCE 损失的温度参数
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.model = base_model
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 将模型移到指定设备
        self.model.to(self.device)

        # 多 GPU 支持: 用 DataParallel 包裹底层 transformer
        self._using_data_parallel = False
        if multi_gpu and self.device.startswith('cuda') and torch.cuda.device_count() > 1:
            transformer = self.model._first_module()
            original_model = transformer.auto_model
            transformer.auto_model = torch.nn.DataParallel(original_model)
            transformer.auto_model.config = original_model.config
            self._using_data_parallel = True

        print(f"[ContrastiveLearner] 初始化完成")
        print(f"  - 模型: {self.model._first_module().__class__.__name__}")
        print(f"  - 设备: {self.device}")
        if self._using_data_parallel:
            print(f"  - 多GPU: DataParallel ({torch.cuda.device_count()} GPUs)")
        print(f"  - 温度参数: {self.temperature}")

    def _unwrap_data_parallel(self):
        """训练结束后解除 DataParallel 包裹，恢复单 GPU 模型用于保存"""
        if self._using_data_parallel:
            transformer = self.model._first_module()
            transformer.auto_model = transformer.auto_model.module
            self._using_data_parallel = False

    def _save_loss_log(self, loss_history: list, epochs: int, data_name: str = "unknown"):
        """将每个 epoch 的 loss 保存为 JSON 文件"""
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)
        log_data = {
            "dataset": data_name,
            "epochs": list(range(1, epochs + 1)),
            "losses": [round(l, 6) for l in loss_history],
        }
        log_path = log_dir / f"{data_name}_loss.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"  Loss 日志已保存: {log_path}")

    def compute_contrastive_loss(self,
                                  embeddings1: torch.Tensor,
                                  embeddings2: torch.Tensor) -> torch.Tensor:
        """
        计算 InfoNCE 对比损失

        Args:
            embeddings1: 第一个视图的嵌入 [batch_size, embedding_dim]
            embeddings2: 第二个视图的嵌入 [batch_size, embedding_dim]

        Returns:
            loss: 标量损失值

        数学公式:
            L = -log( exp(sim(z_i, z_i+) / τ) / Σ_j exp(sim(z_i, z_j) / τ) )
        """
        batch_size = embeddings1.shape[0]

        # 归一化 (使用余弦相似度)
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.mm(embeddings1, embeddings2.t()) / self.temperature

        # labels = [0, 1, 2, ..., batch_size-1] (对角线位置)
        labels = torch.arange(batch_size, device=self.device)

        # InfoNCE 损失
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.t(), labels)

        # 平均两个方向的损失
        loss = (loss_12 + loss_21) / 2

        return loss

    def train_epoch(self,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> float:
        """
        训练一个 epoch

        Args:
            dataloader: 数据加载器
            optimizer: 优化器

        Returns:
            avg_loss: 平均损失值
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)

        for view1_texts, view2_texts in pbar:
            # 使用模型的编码方法,但需要保持梯度
            # tokenize 并移到设备
            view1_features = self.model.tokenize(view1_texts)
            view2_features = self.model.tokenize(view2_texts)

            view1_features = {k: v.to(self.device) for k, v in view1_features.items()}
            view2_features = {k: v.to(self.device) for k, v in view2_features.items()}

            # 通过模型前向传播获取嵌入(保留梯度)
            embeddings1 = self.model(view1_features)['sentence_embedding']
            embeddings2 = self.model(view2_features)['sentence_embedding']

            # 计算对比损失
            loss = self.compute_contrastive_loss(embeddings1, embeddings2)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self,
              entities: List[str],
              epochs: int = 10,
              batch_size: int = 64,
              learning_rate: float = 1e-5,
              warmup_steps: int = 100,
              sample_rate: float = 1.0,
              augmentation_methods1: List[str] = ['mask', 'shuffle'],
              augmentation_methods2: List[str] = ['dropout', 'synonym']) -> SentenceTransformer:
        """
        对比学习训练主流程

        Args:
            entities: 所有实体文本列表
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            warmup_steps: 预热步数
            sample_rate: 采样率 (对于大数据集,可 < 1.0)
            augmentation_methods1: 第一个视图的增强方法
            augmentation_methods2: 第二个视图的增强方法

        Returns:
            finetuned_model: 微调后的模型
        """
        print("\n" + "=" * 60)
        print("开始对比学习训练")
        print("=" * 60)

        # 采样 (对于大数据集)
        if sample_rate < 1.0:
            sample_size = int(len(entities) * sample_rate)
            sampled_entities = np.random.choice(entities, sample_size, replace=False).tolist()
            print(f"采样: {len(sampled_entities)} / {int(len(sampled_entities) / sample_rate)} ({sample_rate * 100:.1f}%)")
            entities = sampled_entities

        print(f"训练配置:")
        print(f"  - 实体数量: {len(entities)}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - 增强方法 (View1): {augmentation_methods1}")
        print(f"  - 增强方法 (View2): {augmentation_methods2}")

        # 创建数据集和数据加载器
        dataset = ContrastiveDataset(
            entities,
            augmentation_methods1=augmentation_methods1,
            augmentation_methods2=augmentation_methods2
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 设为 0 避免多进程问题
            pin_memory=True if self.device == 'cuda' else False
        )

        # 优化器 (AdamW)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # 学习率调度器 (带 warmup)
        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
            else max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        )

        print(f"\n开始训练 (总步数: {total_steps})...\n")

        # 训练循环
        best_loss = float('inf')
        loss_history = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, optimizer)
            loss_history.append(avg_loss)

            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  → 新的最佳损失: {best_loss:.4f}")

        # 保存 loss 日志
        self._save_loss_log(loss_history, epochs)

        print("\n" + "=" * 60)
        print("对比学习训练完成!")
        print(f"最佳损失: {best_loss:.4f}")
        print("=" * 60 + "\n")

        self._unwrap_data_parallel()
        return self.model
    def train_supervised(self,
                         data_path: str,
                         data_name: str,
                         epochs: int = 10,
                         batch_size: int = 64,
                         learning_rate: float = 1e-5,
                         warmup_steps: int = 100) -> SentenceTransformer:
        """
        有监督对比学习训练 — 使用标注正对

        Args:
            data_path: 训练数据根目录
            data_name: 数据集名称
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            warmup_steps: 预热步数

        Returns:
            finetuned_model: 微调后的模型
        """
        print("\n" + "=" * 60)
        print("开始有监督对比学习训练")
        print("=" * 60)

        # 创建数据集和数据加载器
        dataset = SupervisedPairDataset(data_path, data_name)

        print(f"训练配置:")
        print(f"  - 正样本对数量: {len(dataset)}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Warmup steps: {warmup_steps}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        # 优化器 (AdamW)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # 学习率调度器 (带 warmup)
        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
            else max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        )

        print(f"\n开始训练 (总步数: {total_steps})...\n")

        # 训练循环
        best_loss = float('inf')
        loss_history = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, optimizer)
            loss_history.append(avg_loss)

            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  → 新的最佳损失: {best_loss:.4f}")

        # 保存 loss 日志
        self._save_loss_log(loss_history, epochs, data_name=data_name)

        print("\n" + "=" * 60)
        print("有监督对比学习训练完成!")
        print(f"最佳损失: {best_loss:.4f}")
        print("=" * 60 + "\n")

        self._unwrap_data_parallel()
        return self.model


def contrastive_finetune(model: SentenceTransformer,
                         entities: List[str],
                         args,
                         cache_dir: str = "finetuned_models") -> SentenceTransformer:
    """
    便捷函数: 对比学习微调（支持模型缓存）

    供 main.py 调用

    Args:
        model: 预训练的 SentenceTransformer 模型
        entities: 实体文本列表
        args: 参数对象 (来自 args.py)
        cache_dir: 缓存目录

    Returns:
        finetuned_model: 微调后的模型
    """
    import os
    from pathlib import Path

    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # 生成缓存文件名（基于数据集名称、模型类型和关键参数）
    # 添加模型类型到缓存文件名，确保不同模型的缓存分开存储
    # 为了向后兼容，minilm 模型不添加模型类型后缀
    model_type = getattr(args, 'model_type', 'minilm')
    if model_type == "minilm":
        cache_filename = f"{args.data_name}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}"
    else:
        cache_filename = f"{args.data_name}_{model_type}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}"
    cache_model_path = cache_path / cache_filename

    # 检查是否存在缓存模型
    if cache_model_path.exists() and not args.force_retrain:
        print("\n" + "=" * 60)
        print("发现缓存的微调模型!")
        print("=" * 60)
        print(f"  缓存路径: {cache_model_path}")
        print(f"  数据集: {args.data_name}")
        print(f"  跳过训练，直接加载模型...")
        print("=" * 60 + "\n")

        try:
            _load_cached_weights(model, cache_model_path)
            print(f"✅ 成功加载缓存模型权重: {cache_filename}\n")
            return model
        except Exception as e:
            print(f"⚠️  加载缓存模型失败: {e}")
            print("   将重新训练模型...\n")

    # 如果没有缓存或加载失败，进行训练
    print("\n" + "=" * 60)
    print("未找到缓存模型，开始训练...")
    print("=" * 60 + "\n")

    # 初始化学习器
    learner = ContrastiveLearner(
        base_model=model,
        temperature=args.cl_temperature,
        device=args.device,
        multi_gpu=getattr(args, 'multi_gpu', False)
    )

    finetuned_model = learner.train(
        entities=entities,
        epochs=args.cl_epochs,
        batch_size=args.cl_batch_size,
        learning_rate=args.cl_learning_rate,
        sample_rate=args.cl_sample_rate,
        augmentation_methods1=['mask', 'shuffle'],
        augmentation_methods2=['dropout']
    )

    # 保存微调后的模型
    print("\n" + "=" * 60)
    print("保存微调模型到缓存...")
    print("=" * 60)
    try:
        finetuned_model.save(str(cache_model_path))
        print(f"✅ 模型已保存: {cache_model_path}")
        print(f"   下次运行将直接加载此模型（节省 ~15 分钟）")
    except Exception as e:
        print(f"⚠️  保存模型失败: {e}")
    print("=" * 60 + "\n")

    return finetuned_model


def supervised_contrastive_finetune(model: SentenceTransformer,
                                     args,
                                     cache_dir: str = "finetuned_models") -> SentenceTransformer:
    """
    便捷函数: 有监督对比学习微调（支持模型缓存）

    从 llm_training_data/{data_name}/labeled_pairs.json 加载标注正对进行训练。

    Args:
        model: 预训练的 SentenceTransformer 模型
        args: 参数对象 (来自 args.py)
        cache_dir: 缓存目录

    Returns:
        finetuned_model: 微调后的模型
    """
    import os

    # 创建缓存目录
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # 生成缓存文件名（加 _supervised 后缀，与自监督模型分开存储）
    model_type = getattr(args, 'model_type', 'minilm')
    if model_type == "minilm":
        cache_filename = f"{args.data_name}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}_supervised"
    else:
        cache_filename = f"{args.data_name}_{model_type}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}_supervised"
    cache_model_path = cache_path / cache_filename

    # 检查是否存在缓存模型
    if cache_model_path.exists() and not args.force_retrain:
        print("\n" + "=" * 60)
        print("发现缓存的有监督微调模型!")
        print("=" * 60)
        print(f"  缓存路径: {cache_model_path}")
        print(f"  数据集: {args.data_name}")
        print(f"  跳过训练，直接加载模型...")
        print("=" * 60 + "\n")

        try:
            _load_cached_weights(model, cache_model_path)
            print(f"✅ 成功加载缓存模型权重: {cache_filename}\n")
            return model
        except Exception as e:
            print(f"⚠️  加载缓存模型失败: {e}")
            print("   将重新训练模型...\n")

    # 如果没有缓存或加载失败，进行训练
    print("\n" + "=" * 60)
    print("未找到缓存模型，开始有监督对比学习训练...")
    print("=" * 60 + "\n")

    # 初始化学习器
    learner = ContrastiveLearner(
        base_model=model,
        temperature=args.cl_temperature,
        device=args.device,
        multi_gpu=getattr(args, 'multi_gpu', False)
    )

    finetuned_model = learner.train_supervised(
        data_path=args.cl_training_data_dir,
        data_name=args.data_name,
        epochs=args.cl_epochs,
        batch_size=args.cl_batch_size,
        learning_rate=args.cl_learning_rate,
    )

    # 保存微调后的模型
    print("\n" + "=" * 60)
    print("保存有监督微调模型到缓存...")
    print("=" * 60)
    try:
        finetuned_model.save(str(cache_model_path))
        print(f"✅ 模型已保存: {cache_model_path}")
        print(f"   下次运行将直接加载此模型")
    except Exception as e:
        print(f"⚠️  保存模型失败: {e}")
    print("=" * 60 + "\n")

    return finetuned_model


# ==================== 单元测试 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("测试对比学习模块")
    print("=" * 60)

    # 创建测试数据
    test_entities = [
        "Apple iPhone 8 Plus 64GB Silver",
        "Samsung Galaxy S8 64GB Black",
        "Huawei P10 32GB Gold",
        "Apple iPhone 7 32GB Rose Gold",
        "Samsung Galaxy S7 Edge 32GB Silver",
        "Huawei Mate 9 64GB Space Gray",
        "Apple iPhone 6S 16GB Gold",
        "Samsung Galaxy Note 8 64GB Black",
    ] * 4  # 复制几次以增加数据量

    print(f"\n测试数据: {len(test_entities)} 个实体\n")

    # 加载预训练模型
    print("加载预训练模型...")
    model = SentenceTransformer('all-MiniLM-L12-v2')
    print(f"模型加载完成: {model._first_module().__class__.__name__}\n")

    # 创建对比学习训练器
    learner = ContrastiveLearner(
        base_model=model,
        temperature=0.07,
        device='cpu'  # 使用 CPU 进行测试
    )

    # 测试对比损失计算
    print("\n测试 1: 对比损失计算")
    print("-" * 40)
    test_batch = test_entities[:4]
    view1, view2 = [], []
    for entity in test_batch:
        v1, v2 = create_augmented_pair(entity)
        view1.append(v1)
        view2.append(v2)

    embeddings1 = model.encode(view1, convert_to_tensor=True)
    embeddings2 = model.encode(view2, convert_to_tensor=True)

    loss = learner.compute_contrastive_loss(embeddings1, embeddings2)
    print(f"批次大小: {len(test_batch)}")
    print(f"嵌入维度: {embeddings1.shape}")
    print(f"对比损失: {loss.item():.4f}")
    print("✅ 损失计算正常\n")

    # 测试训练流程 (少量 epochs)
    print("\n测试 2: 训练流程 (2 epochs)")
    print("-" * 40)

    finetuned_model = learner.train(
        entities=test_entities,
        epochs=2,
        batch_size=8,
        learning_rate=1e-5,
        augmentation_methods1=['mask'],
        augmentation_methods2=['shuffle']
    )

    print("✅ 训练流程正常\n")

    # 验证模型微调效果 (可选)
    print("\n测试 3: 验证微调效果")
    print("-" * 40)
    test_entity = "Apple iPhone 8 Plus 64GB Silver"
    similar_entity = "Apple iPhone 8 Plus 64 Gigabyte White"  # 相似实体
    different_entity = "Samsung Galaxy S8 64GB Black"  # 不同实体

    emb_test = model.encode(test_entity, convert_to_tensor=True)
    emb_similar = model.encode(similar_entity, convert_to_tensor=True)
    emb_different = model.encode(different_entity, convert_to_tensor=True)

    sim_similar = F.cosine_similarity(emb_test.unsqueeze(0), emb_similar.unsqueeze(0))
    sim_different = F.cosine_similarity(emb_test.unsqueeze(0), emb_different.unsqueeze(0))

    print(f"测试实体: {test_entity}")
    print(f"相似实体: {similar_entity}")
    print(f"  相似度: {sim_similar.item():.4f}")
    print(f"不同实体: {different_entity}")
    print(f"  相似度: {sim_different.item():.4f}")

    if sim_similar > sim_different:
        print("✅ 相似实体的相似度更高 (符合预期)")
    else:
        print("⚠️  可能需要更多训练 epochs")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
