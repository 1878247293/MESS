"""
对比学习训练 — Sentence-BERT 微调。

实现 SimCLR 那一套：InfoNCE + 批内负采样。
从 labeled_pairs.json 读现成的正对作为监督信号。
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file as load_safetensors
from pathlib import Path
from tqdm import tqdm


def _load_cached_weights(model: SentenceTransformer, cache_model_path: Path) -> SentenceTransformer:
    """
    复用已经实例化好的 SentenceTransformer，只把缓存里的权重灌进去。
    避免再走一次 SentenceTransformer(path)，那个加载链路比较慢。
    """
    weights_path = cache_model_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_safetensors(str(weights_path), device=str(model.device))
        model[0].auto_model.load_state_dict(state_dict)
    else:
        # 兜底：老版本可能存的是 .bin
        weights_path = cache_model_path / "pytorch_model.bin"
        state_dict = torch.load(str(weights_path), map_location=model.device)
        model[0].auto_model.load_state_dict(state_dict)
    return model


# 数据集

class SupervisedPairDataset(Dataset):
    """从标注文件读正对"""

    def __init__(self, data_path: str, data_name: str):
        json_file = Path(data_path) / data_name / "labeled_pairs.json"
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        # 两种历史格式都兼容
        # 新版直接是 pairs（全是正对），老版是 triplets，要按 label==1 过一道
        if "pairs" in data:
            self.pairs = [(t[0], t[1]) for t in data["pairs"]]
        else:
            self.pairs = [(t[0], t[1]) for t in data["triplets"] if t[2] == 1]
        print(f"[SupervisedPairDataset] 加载 {len(self.pairs)} 个正对 (from {json_file})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # (text_a, text_b)


class ContrastiveLearner:
    """训练壳，把模型 / 损失 / 多卡 / 日志包到一起"""

    def __init__(self,
                 base_model: SentenceTransformer,
                 temperature: float = 0.07,
                 device: str = 'cuda',
                 multi_gpu: bool = False):
        self.model = base_model
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        # 多卡：用 DataParallel 包底层 transformer
        self._using_data_parallel = False
        if multi_gpu and self.device.startswith('cuda') and torch.cuda.device_count() > 1:
            transformer = self.model._first_module()
            original_model = transformer.auto_model
            transformer.auto_model = torch.nn.DataParallel(original_model)
            transformer.auto_model.config = original_model.config
            self._using_data_parallel = True

        print(f"[ContrastiveLearner] init")
        print(f"  - model: {self.model._first_module().__class__.__name__}")
        print(f"  - device: {self.device}")
        if self._using_data_parallel:
            print(f"  - multi-GPU: DataParallel ({torch.cuda.device_count()} GPUs)")
        print(f"  - temperature: {self.temperature}")

    def _unwrap_data_parallel(self):
        """训完之后还原模型，DataParallel 包着的没法直接 save"""
        if self._using_data_parallel:
            transformer = self.model._first_module()
            transformer.auto_model = transformer.auto_model.module
            self._using_data_parallel = False

    def _save_loss_log(self, loss_history: list, epochs: int, data_name: str = "unknown"):
        """每个 epoch 的 loss 落盘 JSON，方便后面画曲线"""
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
        print(f"  loss 日志: {log_path}")

    def compute_contrastive_loss(self,
                                  embeddings1: torch.Tensor,
                                  embeddings2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE，双向都算一遍取平均。

        L = -log( exp(sim(z_i, z_i+) / tau) / sum_j exp(sim(z_i, z_j) / tau) )
        """
        batch_size = embeddings1.shape[0]

        # 归一化后用余弦相似度
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # [B, B] 的相似度矩阵，正对在对角线上
        similarity_matrix = torch.mm(embeddings1, embeddings2.t()) / self.temperature

        labels = torch.arange(batch_size, device=self.device)

#          exp([8.0, 2.0, 1.0]) = [2980.96, 7.39, 2.72]
#          sum             = 2991.07
#          Z_0 = log(2991.07) ≈ 8.0036
#          nll_0 = Z_0 - S[0,0] = 8.0036 - 8.0 = 0.0036  ← 损失很小,正对得分远高于其他
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.t(), labels)

        loss = (loss_12 + loss_21) / 2

        return loss

    def train_epoch(self,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)#进度条包装

        for text_a_batch, text_b_batch in pbar:
            # 必须保留梯度，所以不能直接走 model.encode
            features_a = self.model.tokenize(text_a_batch)#把文本切成 token ID,留着后面继续走前向(梯度可传)
            features_b = self.model.tokenize(text_b_batch)
            #CPU 搬到 GPU
            features_a = {k: v.to(self.device) for k, v in features_a.items()}
            features_b = {k: v.to(self.device) for k, v in features_b.items()}

            # 前向传播，拿 sentence_embedding
            embeddings_a = self.model(features_a)['sentence_embedding']
            embeddings_b = self.model(features_b)['sentence_embedding']

            loss = self.compute_contrastive_loss(embeddings_a, embeddings_b)
            #清掉上一步累积的梯度
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #优化器按梯度更新权重
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self,
              data_path: str,
              data_name: str,
              epochs: int = 10,
              batch_size: int = 64,
              learning_rate: float = 1e-5,
              warmup_steps: int = 100) -> SentenceTransformer:
        """从 labeled_pairs.json 读正对训练"""
        print("\n" + "=" * 60)
        print("supervised CL")
        print("=" * 60)
        #读正样本对
        dataset = SupervisedPairDataset(data_path, data_name)

        print(f"config:")
        print(f"  - 正对数: {len(dataset)}")
        print(f"  - epochs: {epochs}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - lr: {learning_rate}")
        print(f"  - warmup_steps: {warmup_steps}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
            else max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        )

        print(f"\n开始训练 (total steps = {total_steps})...\n")

        best_loss = float('inf')
        loss_history = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, optimizer)
            loss_history.append(avg_loss)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  -> new best loss: {best_loss:.4f}")

        self._save_loss_log(loss_history, epochs, data_name=data_name)

        print("\n" + "=" * 60)
        print("done")
        print(f"best loss: {best_loss:.4f}")
        print("=" * 60 + "\n")

        self._unwrap_data_parallel()
        return self.model


def contrastive_finetune(model: SentenceTransformer,
                         args,
                         cache_dir: str = "finetuned_models") -> SentenceTransformer:
    """
    有监督对比学习入口。标注正对从 cl_training_data_dir/{data_name}/labeled_pairs.json 读。
    命中缓存就直接载权重，没命中就训练完再保存。
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # 缓存名带数据集 / 模型类型 / 关键超参；保留 _supervised 后缀以兼容历史缓存目录
    model_type = getattr(args, 'model_type', 'minilm')
    if model_type == "minilm":
        cache_filename = f"{args.data_name}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}_supervised"
    else:
        cache_filename = f"{args.data_name}_{model_type}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}_supervised"
    cache_model_path = cache_path / cache_filename

    if cache_model_path.exists() and not args.force_retrain:
        print("\n" + "=" * 60)
        print("命中缓存的微调模型")
        print("=" * 60)
        print(f"  path: {cache_model_path}")
        print(f"  dataset: {args.data_name}")
        print(f"  跳过训练，直接加载...")
        print("=" * 60 + "\n")

        try:
            _load_cached_weights(model, cache_model_path)
            print(f"载入成功: {cache_filename}\n")
            return model
        except Exception as e:
            print(f"载入失败: {e}")
            print("回退到训练流程...\n")

    print("\n" + "=" * 60)
    print("没有缓存，开始训练...")
    print("=" * 60 + "\n")

    learner = ContrastiveLearner(
        base_model=model,
        temperature=args.cl_temperature,
        device=args.device,
        multi_gpu=getattr(args, 'multi_gpu', False)
    )

    finetuned_model = learner.train(
        data_path=args.cl_training_data_dir,
        data_name=args.data_name,
        epochs=args.cl_epochs,
        batch_size=args.cl_batch_size,
        learning_rate=args.cl_learning_rate,
    )

    print("\n" + "=" * 60)
    print("写入缓存...")
    print("=" * 60)
    try:
        finetuned_model.save(str(cache_model_path))
        print(f"saved: {cache_model_path}")
        print(f"  下次同参数运行直接走缓存")
    except Exception as e:
        print(f"save 失败: {e}")
    print("=" * 60 + "\n")

    return finetuned_model
