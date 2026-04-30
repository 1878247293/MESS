import os
from dataclasses import dataclass

# 强制 HuggingFace 相关库进入离线模式，彻底防止网络超时卡顿
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import tyro


@dataclass
class MainArgs:
    data_path: str = "data"  # 修改为实际数据目录
    data_name: str = "Geo"  # 改为 Geo 数据集（已完成网格搜索调优）

    # selecting
    eer_flag: bool = True
    col_sim_threshold: float = 0.8  # gamma - 最优值！网格搜索结果（之前 0.9 仅得 71.99% F1）
    selection_rate: float = 0.2  # r - 用于属性选择阶段的采样率
    manual_selected_attrs: str = ""  # 手工指定属性选择结果，逗号分隔；设置后覆盖自动属性选择
    data_sample_rate: float = 1.0  # 对原始数据集的采样率(用于减少内存占用)
    # merging
    k: int = 1  # k
    min_dis: float = 0.5  # m - 距离阈值（默认值，可被数据集配置覆盖）
    # parallel
    run_in_parallel: bool = False
    multi_gpu: bool = False  # 是否启用多GPU DataParallel（适用于GPU充足的场景）

    # ========== 智能表配对相关参数 ==========
    use_smart_pairing: bool = False  # 是否启用智能表配对
    smart_pairing_strategy: str = "optimal"  # 配对策略: similarity, optimal, complementary

    # ========== 模型选择参数 ==========
    # 模型类型: "minilm" 或 "modernbert"
    # 默认使用本地离线模型（由 setup.sh 下载到 model/ 目录）
    model_type: str = "modernbert"
    lm_model_or_path: str = "model/modernbert"  # 默认优先走本地离线路径，如需在线加载可改为 HF 仓库名
    device: str = "cuda"
    seed: int = 3407
    max_seq_length: int = 64
    batch_size: int = 512


    # ========== 高效匹配参数 ==========
    use_efficient_matching: bool = False  # 是否启用高效匹配（单向搜索+反向验证）

    # ========== 自动配置 ==========
    use_dataset_config: bool = True  # 是否自动加载数据集最优配置

    # ========== 候选集输出参数 ==========
    output_candidates: bool = True  # 是否输出候选集信息
    candidates_output_dir: str = "candidates_output"  # 候选集输出目录

    # ========== PathCL-EM: 对比学习相关参数 ==========
    use_contrastive_learning: bool = False  # 是否启用对比学习微调
    cl_mode: str = "self-supervised"           # 对比学习模式: "self-supervised" 或 "supervised"
    cl_training_data_dir: str = "llm_training_data"  # 有监督训练数据目录
    cl_epochs: int = 10  # 对比学习训练轮数
    cl_batch_size: int = 64  # 对比学习批大小
    cl_temperature: float = 0.07  # InfoNCE 损失的温度参数
    cl_learning_rate: float = 1e-5  # 对比学习学习率
    cl_sample_rate: float = 1.0  # 对比学习采样率 (对于超大数据集,可设置 < 1.0)
    force_retrain: bool = False  # 是否强制重新训练（忽略缓存）
    cl_model_cache_dir: str = "finetuned_models"  # 微调模型缓存目录


def build_main_args():
    args = tyro.cli(MainArgs)

    # ========== 自动加载数据集配置 ==========
    if args.use_dataset_config:
        try:
            from dataset_configs import get_dataset_config

            print(f"\n{'='*60}")
            print(f"🔧 自动配置模式: 正在加载 {args.data_name} 的最优参数...")
            print(f"{'='*60}")

            config = get_dataset_config(args.data_name)

            # 只覆盖未被命令行显式指定的参数
            # 判断是否显式指定的方法：比较当前值与默认值
            defaults = MainArgs()

            if args.eer_flag == defaults.eer_flag:
                args.eer_flag = config.eer_flag
            if args.col_sim_threshold == defaults.col_sim_threshold:
                args.col_sim_threshold = config.col_sim_threshold
            if args.selection_rate == defaults.selection_rate:
                args.selection_rate = config.selection_rate
            if args.k == defaults.k:
                args.k = config.k
            if args.min_dis == defaults.min_dis:
                args.min_dis = config.min_dis


            print(f"\n✓ 已应用 {args.data_name} 的最优配置:")
            print(f"  - col_sim_threshold (γ) = {args.col_sim_threshold}")
            print(f"  - min_dis (m)           = {args.min_dis}")
            print(f"{'='*60}\n")

        except KeyError as e:
            print(f"\n⚠️  警告: {e}")
            print(f"⚠️  将使用默认参数运行\n")
        except ImportError:
            print(f"\n⚠️  警告: 未找到 dataset_configs.py，将使用默认参数\n")

    # ========== 根据 model_type 自动设置模型路径 ==========
    # 统一使用本地 model/ 目录，路径由 setup.sh 预先下载好（与仓库 @model 一致）
    MODEL_MAPPING = {
        "minilm": "model/all-MiniLM-L12-v2",
        "modernbert": "model/modernbert",
        "gte-large": "/home/cjx/PathCL-EM/model/gte-large",
        "modernbert-server": "/home/cjx/PathCL-EM/model/modernbert"
    }

    # ========== 兼容：若给的是 HF cache 风格目录，自动解析到 snapshots/<rev> ==========
    # 说明：仓库里的 model/* 可能是带 blobs/refs/snapshots 的结构，实际可加载目录应为 snapshots/<hash>。
    from pathlib import Path

    def _resolve_hf_snapshot_dir(p: str) -> str:
        try:
            base = Path(p)
            if not base.exists() or not base.is_dir():
                return p
            snaps = base / "snapshots"
            refs_main = base / "refs" / "main"
            if snaps.is_dir():
                # 优先用 refs/main 指向的 revision
                if refs_main.is_file():
                    rev = refs_main.read_text().strip()
                    cand = snaps / rev
                    if cand.is_dir():
                        return str(cand)
                # 否则取 snapshots 下任意一个（通常只有一个）
                subdirs = sorted([d for d in snaps.iterdir() if d.is_dir()])
                if subdirs:
                    return str(subdirs[-1])
            return p
        except Exception:
            return p

    # 如果用户通过命令行指定了 model_type，自动设置 lm_model_or_path
    # 但如果用户同时指定了 lm_model_or_path，则优先使用用户指定的路径
    defaults = MainArgs()
    user_specified_path = args.lm_model_or_path != defaults.lm_model_or_path
    if not user_specified_path and (args.model_type != defaults.model_type or args.lm_model_or_path == defaults.lm_model_or_path):
        # 用户未显式指定路径，且指定了 model_type 或 lm_model_or_path 仍是默认值
        if args.model_type in MODEL_MAPPING:
            args.lm_model_or_path = MODEL_MAPPING[args.model_type]
        else:
            print(f"\n⚠️  警告: 未知的模型类型 '{args.model_type}'")
            print(f"⚠️  可用选项: {list(MODEL_MAPPING.keys())}")
            print(f"⚠️  将使用默认路径: {args.lm_model_or_path}\n")

    # ========== 兼容旧路径（models/ -> model/）==========
    # 历史版本曾把离线模型放在 models/ 下；现在统一为仓库的 model/ 目录。
    # 若用户/旧脚本仍传入 models/... 且该目录不存在，则自动回退到 model/ 目录。
    from pathlib import Path as _Path

    legacy_map = {
        "models/modernbert-embed-base": "model/modernbert",
        "models/all-MiniLM-L12-v2": "model/all-MiniLM-L12-v2",
        "modernbert-embed-base": "model/modernbert",
        "all-MiniLM-L12-v2": "model/all-MiniLM-L12-v2",
        "gte-large": "/home/cjx/PathCL-EM/model/gte-large",
        "gte_large": "/home/cjx/PathCL-EM/model/gte-large",
        "modernbert-server": "/home/cjx/PathCL-EM/model/modernbert",
        "server-modernbert": "/home/cjx/PathCL-EM/model/modernbert",
        "/home/cjx/PathCL-EM/model/gte-large": "/home/cjx/PathCL-EM/model/gte-large",
        "/home/cjx/PathCL-EM/model/modernbert": "/home/cjx/PathCL-EM/model/modernbert",
    }
    if isinstance(args.lm_model_or_path, str):
        p = args.lm_model_or_path
        if p in legacy_map and (not _Path(p).exists() or p.startswith("/home/cjx/PathCL-EM/model/")):
            args.lm_model_or_path = legacy_map[p]

    # 若是 HF cache 风格目录，则解析到 snapshots/<rev>
    args.lm_model_or_path = _resolve_hf_snapshot_dir(args.lm_model_or_path)

    # 打印最终生效的本地路径（避免误以为走线上）
    print(f"\n📦 模型选择: {args.model_type}")
    print(f"   → 加载路径: {args.lm_model_or_path}\n")

    # ========== 根据 model_type 自动优化模型参数 ==========
    # 只在用户未显式指定时才自动调整
    MODEL_PARAMS = {
        "minilm": {
            "max_seq_length": 64,    # MiniLM 最优序列长度
            "batch_size": 512,       # MiniLM 推荐批大小
        },
        "modernbert": {
            "max_seq_length": 256,   # ModernBERT 支持更长序列（最大8192）
            "batch_size": 256,       # 序列更长需要减小批大小
        },
        "modernbert-server": {
            "max_seq_length": 256,
            "batch_size": 256,
        }
    }

    if args.model_type in MODEL_PARAMS:
        recommended_params = MODEL_PARAMS[args.model_type]

        # 只在用户未显式指定时才应用推荐参数
        params_changed = []
        if args.max_seq_length == defaults.max_seq_length:
            old_val = args.max_seq_length
            args.max_seq_length = recommended_params["max_seq_length"]
            params_changed.append(f"max_seq_length: {old_val} → {args.max_seq_length}")

        if args.batch_size == defaults.batch_size:
            old_val = args.batch_size
            args.batch_size = recommended_params["batch_size"]
            params_changed.append(f"batch_size: {old_val} → {args.batch_size}")

        if params_changed:
            print(f"🔧 自动优化 {args.model_type} 参数:")
            for change in params_changed:
                print(f"   - {change}")
            print()

    return args
