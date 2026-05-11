"""把 entity groups 拍成正样本对，写到 labeled_pairs.json。"""

import json
import random
import itertools
from datetime import datetime
from pathlib import Path

from .config import DatasetConfig


def format_record_text(record: dict, config: DatasetConfig) -> str:
    """单个 variant -> 拼成一行文本

    例：Geo 的 "name: Brno (Czech Republic), longtitude: 16.608, latitude: 49.195"
    """
    parts = []
    for field in config.text_fields:
        val = record.get(field, "")
        if val is None:
            val = ""
        parts.append(f"{field}: {val}")
    return ", ".join(parts)


def build_positive_pairs(entity_groups: list, config: DatasetConfig) -> list:
    """组内任两个 variant 配成一对"""
    pairs = []
    for group in entity_groups:
        variants = group.get("variants", [])
        texts = [format_record_text(v, config) for v in variants]
        for i, j in itertools.combinations(range(len(texts)), 2):
            pairs.append([texts[i], texts[j], 1])
    return pairs


def build_pairs(entity_groups: list, config: DatasetConfig,
                seed: int = 42) -> tuple:
    """只产正样本，负样本交给 in-batch negatives"""
    positive_pairs = build_positive_pairs(entity_groups, config)
    num_positive = len(positive_pairs)

    rng = random.Random(seed)
    rng.shuffle(positive_pairs)

    total_variants = sum(len(g.get("variants", [])) for g in entity_groups)
    avg_variants = total_variants / len(entity_groups) if entity_groups else 0

    metadata = {
        "dataset": config.name,
        "method": "llm_generated",
        "description": f"LLM 生成的正样本对，负样本走 in-batch negatives。{config.description}",
        "num_entity_groups": len(entity_groups),
        "variants_per_entity": round(avg_variants),
        "num_pairs": num_positive,
        "num_positive": num_positive,
        "generation_time": datetime.now().isoformat(),
    }

    print(f"  pairs: {num_positive}")

    return positive_pairs, metadata


def save_output(entity_groups: list, pairs: list,
                metadata: dict, output_path: str):
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
    print(f"  saved: {path} ({size_mb:.2f} MB)")
