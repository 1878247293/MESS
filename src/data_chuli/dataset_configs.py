"""
数据集特定的最优超参数配置

本文件记录了每个数据集经过网格搜索找到的最优参数组合。
当运行 main.py 时，会根据 data_name 自动加载对应的配置。

配置来源：
- Geo: 网格搜索结果（2025-11-11）grid_search_summary.md
- Music-20: 网格搜索结果（2025-11-11）✅ F1=90.15%
- Music-200: 网格搜索 + 验证（2025-11-12）✅ F1=82.41%（已达到论文水平）
- Music-2000: 待网格搜索确定
- Shopee: 网格搜索结果（2025-12-07）✅ F1=28.79%（超越论文 26.2%）
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class DatasetConfig:
    """单个数据集的配置"""
    # 属性选择参数
    eer_flag: bool = True
    col_sim_threshold: float = 0.8
    selection_rate: float = 0.2

    # 合并参数
    k: int = 1
    min_dis: float = 0.5

    # 剪枝参数
    eps: float = 1.0

    # PathCL-EM 对比学习参数
    use_contrastive_learning: bool = False
    cl_epochs: int = 10
    cl_batch_size: int = 64
    cl_temperature: float = 0.07
    cl_learning_rate: float = 1e-5
    cl_sample_rate: float = 1.0

    # 元数据（用于记录）
    description: str = ""
    f1_score: float = 0.0  # 该配置达到的 F1 分数
    search_date: str = ""  # 网格搜索日期


# ========== 各数据集的最优配置 ==========

# Geo 数据集 - 已完成网格搜索
GEO_CONFIG = DatasetConfig(
    # 基础参数（网格搜索结果）
    eer_flag=True,
    col_sim_threshold=0.8,  # 最关键！0.9→71.99%, 0.8→90.87%
    selection_rate=0.2,
    k=1,
    min_dis=0.5,  # 最优值
    eps=1.0,      # 最优值

    # PathCL-EM 增强配置（可选）
    use_contrastive_learning=False,

    # 元数据
    description="Geo 数据集最优配置（网格搜索）",
    f1_score=90.87,
    search_date="2025-11-11"
)

# Geo + PathCL-EM 配置
GEO_PATHCL_CONFIG = DatasetConfig(
    # 继承基础最优参数
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,
    k=1,
    min_dis=0.5,
    eps=1.0,

    # 启用 PathCL-EM
    use_contrastive_learning=True,
    cl_epochs=10,
    cl_batch_size=64,
    cl_temperature=0.07,
    cl_learning_rate=1e-5,
    cl_sample_rate=1.0,

    # 元数据
    description="Geo 数据集 + PathCL-EM 增强",
    f1_score=91.88,
    search_date="2025-11-11"
)

# Geo + 对比学习配置（推荐测试）⭐
GEO_HARD_NEG_CONFIG = DatasetConfig(
    # 基础最优参数
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,
    k=1,
    min_dis=0.5,
    eps=1.0,

    # 启用对比学习
    use_contrastive_learning=True,
    cl_epochs=10,
    cl_batch_size=64,
    cl_temperature=0.07,
    cl_learning_rate=1e-5,
    cl_sample_rate=1.0,

    # 元数据
    description="Geo + 对比学习（推荐）",
    f1_score=0.0,  # 待测试
    search_date="2025-12-07"
)

# Music-20 数据集 - 已完成网格搜索 ✅
MUSIC20_CONFIG = DatasetConfig(
    # 网格搜索最优配置
    eer_flag=True,
    col_sim_threshold=0.9,  # 关键! 0.8→45.14%, 0.9→90.15%
    selection_rate=0.2,
    k=1,
    min_dis=0.35,  # 最优值 (比 Geo 的 0.5 更小)
    eps=0.8,       # 0.8 vs 1.0 影响很小

    # PathCL-EM 暂时禁用（基线配置）
    use_contrastive_learning=False,

    # 元数据
    description="Music-20 数据集最优配置（网格搜索）",
    f1_score=90.15,  # 超越论文 88.6%!
    search_date="2025-11-11"
)

# Music-20 + 对比学习配置（推荐）✨
MUSIC20_CONFIG_CL = DatasetConfig(
    # 继承基础最优参数
    eer_flag=True,
    col_sim_threshold=0.9,
    selection_rate=0.2,
    k=1,
    min_dis=0.35,
    eps=0.8,

    # 启用对比学习（提升 +3.07%）
    use_contrastive_learning=True,  # ✅ 启用
    cl_epochs=10,
    cl_batch_size=64,
    cl_temperature=0.07,
    cl_learning_rate=1e-5,
    cl_sample_rate=1.0,

    # 元数据
    description="Music-20 + 对比学习（推荐配置）",
    f1_score=93.22,  # 相比基线 90.15% 提升了 +3.07%!
    search_date="2025-11-12"
)

# Music-200 数据集 - 已完成验证 ✅
MUSIC200_CONFIG = DatasetConfig(
    # 网格搜索 + 验证最优配置 (2025-11-12)
    eer_flag=True,
    col_sim_threshold=0.9,  # 网格搜索最优值
    selection_rate=0.2,     # ⚠️ 关键！论文推荐值，不是 0.1！
    k=1,
    min_dis=0.35,           # 网格搜索最优值
    eps=0.8,                # 测试更严格的剪枝参数（从 1.0 改为 0.8）

    use_contrastive_learning=False,

    description="Music-200 最优配置（网格搜索 + 验证，与论文一致）",
    f1_score=82.41,  # 论文: 82.2%，已验证！
    search_date="2025-11-12"
)

# Music-2000 数据集 - 待优化
MUSIC2000_CONFIG = DatasetConfig(
    # 临时配置（继承 Music-200）
    eer_flag=True,
    col_sim_threshold=0.8,
    selection_rate=0.2,  # 论文推荐值（<5M实体的数据集）
    k=1,
    min_dis=0.3,
    eps=0.8,

    use_contrastive_learning=False,
    cl_sample_rate=0.3,  # 超大数据集需要采样

    description="Music-2000 临时配置（待网格搜索优化）",
    f1_score=0.0,
    search_date="待定"
)

# Shopee 数据集 - 已完成网格搜索 ✅
SHOPEE_CONFIG = DatasetConfig(
    # 网格搜索最优配置 (2025-12-07)
    eer_flag=True,
    col_sim_threshold=0.9,  # 网格搜索最优值（0.8 vs 0.9 影响不大）
    selection_rate=0.2,
    k=1,
    min_dis=0.5,            # 最优值！（Shopee 需要更严格的距离阈值）
    eps=1.0,                # 最优值（0.8 vs 1.0 影响很小）

    use_contrastive_learning=False,

    description="Shopee 最优配置（网格搜索，电商产品匹配）",
    f1_score=28.79,  # F1: 28.79%, pair-F1: 40.78%（超越论文 26.2%）
    search_date="2025-12-07"
)


# ========== 配置查找表 ==========
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    # Geo
    "Geo": GEO_CONFIG,
    "geo": GEO_CONFIG,
    "Geo-PathCL": GEO_PATHCL_CONFIG,
    "Geo-HardNeg": GEO_HARD_NEG_CONFIG,

    # Music 系列
    "Music-20": MUSIC20_CONFIG,
    "music-20": MUSIC20_CONFIG,
    "Music-20-CL": MUSIC20_CONFIG_CL,  # 对比学习增强版（推荐）
    "Music-200": MUSIC200_CONFIG,
    "music-200": MUSIC200_CONFIG,
    "Music-2000": MUSIC2000_CONFIG,
    "music-2000": MUSIC2000_CONFIG,

    # Shopee
    "Shopee": SHOPEE_CONFIG,
    "shopee": SHOPEE_CONFIG,
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    根据数据集名称获取最优配置

    Args:
        dataset_name: 数据集名称（如 "Geo", "Music-20"）

    Returns:
        该数据集的最优配置（DatasetConfig 对象）

    Raises:
        KeyError: 如果数据集不存在
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(sorted(set(DATASET_CONFIGS.keys())))
        raise KeyError(
            f"未找到数据集 '{dataset_name}' 的配置。\n"
            f"可用的数据集: {available}\n"
            f"如果这是新数据集，请先在 dataset_configs.py 中添加配置。"
        )

    config = DATASET_CONFIGS[dataset_name]
    print(f"✓ 加载数据集配置: {dataset_name}")
    print(f"  描述: {config.description}")
    if config.f1_score > 0:
        print(f"  已知最佳 F1: {config.f1_score:.2f}%")
    if config.search_date != "待定":
        print(f"  配置日期: {config.search_date}")

    return config


def list_all_configs():
    """列出所有可用的数据集配置"""
    print("\n" + "="*60)
    print("所有可用的数据集配置:")
    print("="*60)

    seen = {}
    for name, config in DATASET_CONFIGS.items():
        # 跳过重复的（大小写变体）- 使用 id 作为唯一标识
        config_id = id(config)
        if config_id in seen:
            continue
        seen[config_id] = name

        print(f"\n【{name}】")
        print(f"  min_dis={config.min_dis}, eps={config.eps}, γ={config.col_sim_threshold}")
        print(f"  对比学习: {'启用' if config.use_contrastive_learning else '禁用'}")
        if config.f1_score > 0:
            print(f"  F1 分数: {config.f1_score:.2f}%")
        print(f"  状态: {config.description}")

    print("\n" + "="*60)


if __name__ == "__main__":
    # 测试代码
    list_all_configs()

    print("\n\n测试获取配置:")
    config = get_dataset_config("Geo")
    print(f"  min_dis={config.min_dis}")
    print(f"  eps={config.eps}")
    print(f"  col_sim_threshold={config.col_sim_threshold}")
