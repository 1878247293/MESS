"""
按数据集存的最优超参。
main.py 启动时根据 data_name 自动套，命令行显式给的参数不会被覆盖。
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class DatasetConfig:
    # 属性选择
    eer_flag: bool = True
    col_sim_threshold: float = 0.8
    selection_rate: float = 0.2

    # 合并
    k: int = 1
    min_dis: float = 0.5

    # 剪枝
    eps: float = 1.0

    # 对比学习
    use_contrastive_learning: bool = False
    cl_epochs: int = 10
    cl_batch_size: int = 64
    cl_temperature: float = 0.07
    cl_learning_rate: float = 1e-5
    cl_sample_rate: float = 1.0

    # 留个备注 + 当时的 F1
    description: str = ""
    f1_score: float = 0.0
    search_date: str = ""


# Geo
GEO_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.8,  # 这个最敏感，从 0.9 调到 0.8 才对
    selection_rate=0.2,
    k=1,
    min_dis=0.5,
    eps=1.0,

    use_contrastive_learning=False,

    description="Geo baseline",
    f1_score=90.87,
    search_date="2025-11-11"
)

# Geo + 对比学习
GEO_CL_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,
    k=1,
    min_dis=0.5,
    eps=1.0,

    use_contrastive_learning=True,
    cl_epochs=10,
    cl_batch_size=64,
    cl_temperature=0.07,
    cl_learning_rate=1e-5,
    cl_sample_rate=1.0,

    description="Geo + contrastive learning",
    f1_score=91.88,
    search_date="2025-11-11"
)

# Geo + 难负样本
GEO_HARD_NEG_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,
    k=1,
    min_dis=0.5,
    eps=1.0,

    use_contrastive_learning=True,
    cl_epochs=10,
    cl_batch_size=64,
    cl_temperature=0.07,
    cl_learning_rate=1e-5,
    cl_sample_rate=1.0,

    description="Geo + hard negatives",
    f1_score=0.0,
    search_date="2025-12-07"
)

# Music-20
MUSIC20_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.9,  # 和 Geo 不一样，Music 上 0.9 才好
    selection_rate=0.2,
    k=1,
    min_dis=0.35,  # 比 Geo 更紧
    eps=0.8,

    use_contrastive_learning=False,

    description="Music-20 baseline",
    f1_score=90.15,
    search_date="2025-11-11"
)

# Music-20 + CL
MUSIC20_CONFIG_CL = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.9,
    selection_rate=0.2,
    k=1,
    min_dis=0.35,
    eps=0.8,

    # 加 CL 大约 +3pt
    use_contrastive_learning=True,
    cl_epochs=10,
    cl_batch_size=64,
    cl_temperature=0.07,
    cl_learning_rate=1e-5,
    cl_sample_rate=1.0,

    description="Music-20 + contrastive learning",
    f1_score=93.22,
    search_date="2025-11-12"
)

# Music-200
MUSIC200_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.9,
    selection_rate=0.2,     # 注意：是 0.2 不是 0.1，0.1 会偏
    k=1,
    min_dis=0.35,
    eps=0.8,                # 比 1.0 紧一点

    use_contrastive_learning=False,

    description="Music-200 baseline",
    f1_score=82.41,
    search_date="2025-11-12"
)

# Music-2000：还没正式调
MUSIC2000_CONFIG = DatasetConfig(
    # 暂用 Music-200 的参数
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,
    k=1,
    min_dis=0.3,
    eps=0.8,

    use_contrastive_learning=False,
    cl_sample_rate=0.3,  # 数据量太大，CL 这边采点样

    description="Music-2000 (placeholder)",
    f1_score=0.0,
    search_date="待定"
)

# Shopee
SHOPEE_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.9,  # 0.8 / 0.9 差别不大
    selection_rate=0.2,
    k=1,
    min_dis=0.5,            # Shopee 上要更紧
    eps=1.0,

    use_contrastive_learning=False,

    description="Shopee baseline",
    f1_score=28.79,
    search_date="2025-12-07"
)


# name -> config
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    # Geo
    "Geo": GEO_CONFIG,
    "geo": GEO_CONFIG,
    "Geo-CL": GEO_CL_CONFIG,
    "Geo-HardNeg": GEO_HARD_NEG_CONFIG,

    # Music
    "Music-20": MUSIC20_CONFIG,
    "music-20": MUSIC20_CONFIG,
    "Music-20-CL": MUSIC20_CONFIG_CL,
    "Music-200": MUSIC200_CONFIG,
    "music-200": MUSIC200_CONFIG,
    "Music-2000": MUSIC2000_CONFIG,
    "music-2000": MUSIC2000_CONFIG,

    # Shopee
    "Shopee": SHOPEE_CONFIG,
    "shopee": SHOPEE_CONFIG,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(sorted(set(DATASET_CONFIGS.keys())))
        raise KeyError(
            f"未找到数据集 '{dataset_name}' 的配置。\n"
            f"可用的: {available}\n"
            f"如果是新数据集，请先在 dataset_configs.py 里加一个。"
        )

    config = DATASET_CONFIGS[dataset_name]
    print(f"加载数据集配置: {dataset_name}")
    print(f"  desc: {config.description}")
    if config.f1_score > 0:
        print(f"  历史最佳 F1: {config.f1_score:.2f}%")
    if config.search_date != "待定":
        print(f"  日期: {config.search_date}")

    return config


def list_all_configs():
    print("\n" + "="*60)
    print("可用的数据集配置:")
    print("="*60)

    seen = {}
    for name, config in DATASET_CONFIGS.items():
        # 同一对象会被多个 key 指（比如大小写），去重
        config_id = id(config)
        if config_id in seen:
            continue
        seen[config_id] = name

        print(f"\n[{name}]")
        print(f"  min_dis={config.min_dis}, eps={config.eps}, gamma={config.col_sim_threshold}")
        print(f"  CL: {'on' if config.use_contrastive_learning else 'off'}")
        if config.f1_score > 0:
            print(f"  F1: {config.f1_score:.2f}%")
        print(f"  desc: {config.description}")

    print("\n" + "="*60)


if __name__ == "__main__":
    list_all_configs()

    print("\n\nget_dataset_config('Geo'):")
    config = get_dataset_config("Geo")
    print(f"  min_dis={config.min_dis}")
    print(f"  eps={config.eps}")
    print(f"  col_sim_threshold={config.col_sim_threshold}")
