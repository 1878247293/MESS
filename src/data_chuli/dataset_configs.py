"""
Best hyperparameters stored per dataset.
main.py applies them automatically by data_name at startup; parameters explicitly given on the command line are not overridden.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class DatasetConfig:
    # attribute selection
    eer_flag: bool = True
    col_sim_threshold: float = 0.8
    selection_rate: float = 0.2

    # merging
    k: int = 1
    min_dis: float = 0.5

    # pruning
    eps: float = 1.0

    # contrastive learning
    use_contrastive_learning: bool = False
    cl_epochs: int = 10
    cl_batch_size: int = 64
    cl_temperature: float = 0.07
    cl_learning_rate: float = 1e-5
    cl_sample_rate: float = 1.0

    # a note + the F1 at that time
    description: str = ""
    f1_score: float = 0.0
    search_date: str = ""


# Geo
GEO_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=0.8,  # the most sensitive one; had to drop from 0.9 to 0.8
    selection_rate=0.2,
    k=1,
    min_dis=0.7,
    eps=1.0,

    use_contrastive_learning=False,

    description="Geo baseline",
    f1_score=90.87,
    search_date="2025-11-11"
)

# Geo + contrastive learning
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

# Geo + hard negatives
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
    col_sim_threshold=0.9,  # different from Geo; 0.9 works better on Music
    selection_rate=0.2,
    k=1,
    min_dis=0.35,  # tighter than Geo
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

    # adding CL gives about +3pt
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
    selection_rate=0.2,     # note: it is 0.2, not 0.1; 0.1 drifts
    k=1,
    min_dis=0.35,
    eps=0.8,                # a bit tighter than 1.0

    use_contrastive_learning=False,

    description="Music-200 baseline",
    f1_score=82.41,
    search_date="2025-11-12"
)

# Music-2000: not formally tuned yet
MUSIC2000_CONFIG = DatasetConfig(
    # temporarily using Music-200 parameters
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,
    k=1,
    min_dis=0.3,
    eps=0.8,

    use_contrastive_learning=False,
    cl_sample_rate=0.3,  # dataset is too large, so subsample for CL

    description="Music-2000 (placeholder)",
    f1_score=0.0,
    search_date="TBD"
)

# Shopee
SHOPEE_CONFIG = DatasetConfig(
    eer_flag=True,
    col_sim_threshold=1.0,
    selection_rate=0.2,
    k=1,
    min_dis=0.3,
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
            f"No configuration found for dataset '{dataset_name}'.\n"
            f"Available: {available}\n"
            f"If this is a new dataset, please add one in dataset_configs.py first."
        )

    config = DATASET_CONFIGS[dataset_name]
    print(f"Loading dataset config: {dataset_name}")
    print(f"  desc: {config.description}")
    if config.f1_score > 0:
        print(f"  best historical F1: {config.f1_score:.2f}%")
    if config.search_date != "TBD":
        print(f"  date: {config.search_date}")

    return config


def list_all_configs():
    print("\n" + "="*60)
    print("Available dataset configs:")
    print("="*60)

    seen = {}
    for name, config in DATASET_CONFIGS.items():
        # the same object can be referenced by multiple keys (e.g. different case); dedup
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
