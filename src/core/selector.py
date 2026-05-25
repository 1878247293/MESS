"""
属性选择（EER, Embedding Equivalence Removal）。

主流程的第一阶段。目的：剔除对实体身份没影响的列（如随机 ID），只保留真正能区分
实体的属性，减小后续 KNN 搜索的噪声。

做法：对每个候选列，把该列整列打乱后重新编码整张表，与原始编码做逐行 cosine。
如果均值相似度 `mean_sim ≤ col_sim_threshold`，说明这列被打乱后行向量变化大，
对身份重要，保留；否则丢弃。结果可以被 selector_cache 缓存复用。
"""

from typing import List
import os

# 在 import HF / SentenceTransformers 之前先把离线开关打开
# 否则属性选择阶段偶尔会偷偷探测网络
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from args import MainArgs
from data import textify_table
from log import log
from utils import element_wise_cosine_sim


def auto_selection(tables_df: List[pd.DataFrame], args: MainArgs):
    table_df = pd.concat(tables_df, axis=0)#表纵向拼接
    table_df = table_df.sample(frac=args.selection_rate)

    model = SentenceTransformer(
        args.lm_model_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.max_seq_length = args.max_seq_length
    model.to(args.device)
    #每行 DataFrame 拼成一句话
    sentences_before = textify_table(table_df)
    table_embeddings = model.encode(
        sentences_before,
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )

    selected_attrs = ["tid"]
    attribute_scores = {}

    for name, col in table_df.items():
        if name == "tid":
            continue

        col_copy = col.copy(deep=True)
        table_df[name] = col_copy.sample(frac=1).reset_index(drop=True)
        sentences_after = textify_table(table_df)
        table_df[name] = col

        table_embeddings_after = model.encode(
            sentences_after,
            show_progress_bar=True,
            batch_size=args.batch_size,
            normalize_embeddings=True,
        )
        sim = element_wise_cosine_sim(table_embeddings, table_embeddings_after)#原向量 a · 打乱后向量 b 
        mean_sim = np.mean(sim)#求均值
        attribute_scores[name] = float(mean_sim)

        if mean_sim <= args.col_sim_threshold:
            selected_attrs.append(name)

        log(f"col: {name}, sim: {mean_sim}")

    return selected_attrs, attribute_scores
