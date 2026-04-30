"""格式化输出：构建正样本对并保存 labeled_pairs.json。"""

import json
import random
import itertools
from datetime import datetime
from pathlib import Path

from .config import DatasetConfig


def format_record_text(record: dict, config: DatasetConfig) -> str:
    """将 variant 记录转为文本字符串。

    例如 Geo: "name: Brno (Czech Republic), longtitude: 16.608, latitude: 49.195"
    """
    parts = []
    for field in config.text_fields:
        val = record.get(field, "")
        if val is None:
            val = ""
        parts.append(f"{field}: {val}")
    return ", ".join(parts)


def build_positive_pairs(entity_groups: list, config: DatasetConfig) -> list:
    """构建正样本对：每个 entity group 内所有 variant 组合。

    Returns:
        list of [text_a, text_b, 1]
    """
    pairs = []
    for group in entity_groups:
        variants = group.get("variants", [])
        texts = [format_record_text(v, config) for v in variants]
        for i, j in itertools.combinations(range(len(texts)), 2):
            pairs.append([texts[i], texts[j], 1])
    return pairs


def build_pairs(entity_groups: list, config: DatasetConfig,
                seed: int = 42) -> tuple:
    """构建正样本对（负样本由对比学习的 in-batch negatives 隐式提供）。

    Returns:
        (pairs_list, metadata_dict)
    """
    # 正样本
    positive_pairs = build_positive_pairs(entity_groups, config)
    num_positive = len(positive_pairs)

    # 打乱顺序
    rng = random.Random(seed)
    rng.shuffle(positive_pairs)

    # 计算 variants_per_entity 的平均值
    total_variants = sum(len(g.get("variants", [])) for g in entity_groups)
    avg_variants = total_variants / len(entity_groups) if entity_groups else 0

    metadata = {
        "dataset": config.name,
        "method": "llm_generated",
        "description": f"LLM 自动分析+生成的训练数据（仅正样本对，负样本由 in-batch negatives 提供）。{config.description}",
        "num_entity_groups": len(entity_groups),
        "variants_per_entity": round(avg_variants),
        "num_pairs": num_positive,
        "num_positive": num_positive,
        "generation_time": datetime.now().isoformat(),
    }

    print(f"  正样本对统计:")
    print(f"    正样本对: {num_positive}")

    return positive_pairs, metadata


def save_output(entity_groups: list, pairs: list,
                metadata: dict, output_path: str):
    """保存为 labeled_pairs.json。"""
    output = {
        "metadata": metadata,
        "entity_groups": entity_groups,
        "pairs": pairs,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  已保存: {path} ({size_mb:.2f} MB)")
