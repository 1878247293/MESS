"""
独立的对比学习训练脚本

例:
    # 标准训练
    python train_contrastive.py --data-name Geo --cl-epochs 10

    # 换模型
    python train_contrastive.py --data-name Shopee --model-type minilm

    # 强制重训
    python train_contrastive.py --data-name Geo --force-retrain
"""

from pathlib import Path
import os
import sys
import time

# 必须在 import transformers / sentence_transformers 前先把离线模式打开
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 把 src/ 子目录塞进 sys.path
_base_path = Path(__file__).resolve().parent #获取目录
sys.path.append(str(_base_path / 'src'))
for _sub in ['core', 'llm', 'data_chuli', 'training', 'utils']:
    sys.path.append(str(_base_path / 'src' / _sub))

from sentence_transformers import SentenceTransformer
import torch

# 30/40 系卡上开 TF32
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from args import build_main_args
from contrastive_learning import contrastive_finetune


def main():
    args = build_main_args()#读取并且分析命令行参数

    # 1. 概览
    print(f"\n{'='*60}")
    print(f"独立对比学习训练")
    print(f"{'='*60}")
    print(f"  数据集: {args.data_name}")
    print(f"  模型: {args.model_type} ({args.lm_model_or_path})")
    print(f"  Epochs: {args.cl_epochs}")
    print(f"  Batch size: {args.cl_batch_size}")
    print(f"  Learning rate: {args.cl_learning_rate}")
    print(f"  Temperature: {args.cl_temperature}")
    print(f"  强制重训: {args.force_retrain}")
    print(f"{'='*60}\n")

    print(f"训练数据: {args.cl_training_data_dir}/{args.data_name}/labeled_pairs.json\n")

    # 2. 加载模型
    print("加载 SentenceTransformer 模型...")
    model = SentenceTransformer(args.lm_model_or_path, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    model.to(args.device)
    print(f"  模型已加载到 {args.device}\n")

    # 3. 训练
    print("开始训练...")
    t_train = time.time()

    model = contrastive_finetune(
        model, args, cache_dir=args.cl_model_cache_dir
    )

    train_time = time.time() - t_train
    print(f"  训练耗时: {train_time:.2f}s\n")

    # 4. 摘要
    print(f"{'='*60}")
    print(f"训练摘要")
    print(f"{'='*60}")
    print(f"  数据集: {args.data_name}")
    print(f"  模型: {args.model_type}")
    print(f"  训练耗时: {train_time:.2f}s")
    print(f"  模型缓存: {args.cl_model_cache_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
