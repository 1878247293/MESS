"""
独立的对比学习训练脚本

例:
    # 自监督
    python train_contrastive.py --data-name Geo --cl-mode self-supervised --cl-epochs 10

    # 有监督
    python train_contrastive.py --data-name Geo --cl-mode supervised --cl-epochs 10

    # 换模型
    python train_contrastive.py --data-name Shopee --model-type minilm --cl-mode supervised

    # 强制重训
    python train_contrastive.py --data-name Geo --cl-mode supervised --force-retrain
"""

from itertools import chain
from pathlib import Path
import os
import sys
import time

# 必须在 import transformers / sentence_transformers 前先把离线模式打开
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 把 src/ 子目录塞进 sys.path
_base_path = Path(__file__).resolve().parent
sys.path.append(str(_base_path / 'src'))
for _sub in ['core', 'llm', 'data_chuli', 'training', 'utils']:
    sys.path.append(str(_base_path / 'src' / _sub))

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 30/40 系卡上开 TF32
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from args import build_main_args
from data import read_all_tables, textify_table
from contrastive_learning import contrastive_finetune, supervised_contrastive_finetune


def main():
    args = build_main_args()

    data_path = Path(args.data_path)
    full_data_path = data_path / args.data_name

    # 1. 读数据
    print(f"\n{'='*60}")
    print(f"独立对比学习训练")
    print(f"{'='*60}")
    print(f"  数据集: {args.data_name}")
    print(f"  模式: {args.cl_mode}")
    print(f"  模型: {args.model_type} ({args.lm_model_or_path})")
    print(f"  Epochs: {args.cl_epochs}")
    print(f"  Batch size: {args.cl_batch_size}")
    print(f"  Learning rate: {args.cl_learning_rate}")
    print(f"  Temperature: {args.cl_temperature}")
    print(f"  强制重训: {args.force_retrain}")
    print(f"{'='*60}\n")

    print("加载数据集...")
    t0 = time.time()
    all_sentences = None
    if args.cl_mode == "supervised":
        # 有监督直接读 triplets，不用过原表
        T = 0
        print(f"  有监督模式: 跳过原始数据集，训练数据来自 {args.cl_training_data_dir}/")
    else:
        T, tables_df = read_all_tables(full_data_path)
        table_sentences = [textify_table(table) for table in tables_df]
        all_sentences = list(chain(*table_sentences))
        print(f"  表数量: {T}")
        print(f"  总实体数: {len(all_sentences)}")
    print(f"  耗时: {time.time() - t0:.2f}s\n")

    # 2. 加载模型
    print("加载 SentenceTransformer 模型...")
    trust_code = "modernbert" in str(args.lm_model_or_path).lower()
    model = SentenceTransformer(args.lm_model_or_path, trust_remote_code=trust_code)
    model.max_seq_length = args.max_seq_length
    model.to(args.device)
    print(f"  模型已加载到 {args.device}\n")

    # 3. 训练
    print("开始训练...")
    t_train = time.time()

    if args.cl_mode == "supervised":
        print(f"  模式: supervised (数据目录: {args.cl_training_data_dir})")
        model = supervised_contrastive_finetune(
            model, args, cache_dir=args.cl_model_cache_dir
        )
    else:
        print(f"  模式: self-supervised (数据增强)")
        model = contrastive_finetune(
            model, all_sentences, args, cache_dir=args.cl_model_cache_dir
        )

    train_time = time.time() - t_train
    print(f"  训练耗时: {train_time:.2f}s\n")

    # 4. 摘要
    print(f"{'='*60}")
    print(f"训练摘要")
    print(f"{'='*60}")
    print(f"  数据集: {args.data_name}")
    print(f"  模式: {args.cl_mode}")
    print(f"  模型: {args.model_type}")
    print(f"  训练耗时: {train_time:.2f}s")
    print(f"  模型缓存: {args.cl_model_cache_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
