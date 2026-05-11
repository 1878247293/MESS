import os
from dataclasses import dataclass

# 这两个变量必须在 import HF / sentence-transformers 之前设置
# 否则就会有偶发的网络探测，慢得难受
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import tyro


@dataclass
class MainArgs:
    data_path: str = "data"
    data_name: str = "Geo"  # 默认走 Geo，那个数据集已经调过参

    # selecting
    eer_flag: bool = True
    col_sim_threshold: float = 0.8  # gamma；网格搜出来 0.8 是 Geo 上最好的
    selection_rate: float = 0.2  # r
    manual_selected_attrs: str = ""  # 手工指定属性，逗号分隔；非空时跳过自动选择
    data_sample_rate: float = 1.0  # 大数据集可以下采样
    # merging
    k: int = 1
    min_dis: float = 0.5  # m，距离阈值
    # parallel
    run_in_parallel: bool = False
    multi_gpu: bool = False  # 多卡 DataParallel

    # 智能表配对
    use_smart_pairing: bool = False
    smart_pairing_strategy: str = "optimal"  # similarity / optimal / complementary

    # 模型
    # minilm 或 modernbert，路径默认本地
    model_type: str = "modernbert"
    lm_model_or_path: str = "model/modernbert"  # 离线路径；要在线就改成 HF repo
    device: str = "cuda"
    seed: int = 3407
    max_seq_length: int = 64
    batch_size: int = 512


    # 高效匹配开关
    use_efficient_matching: bool = False  # 单向搜索 + 反向验证

    # 是否自动套用每个数据集的最优参数
    use_dataset_config: bool = True

    # 候选集
    output_candidates: bool = True
    candidates_output_dir: str = "candidates_output"

    # 对比学习
    use_contrastive_learning: bool = False
    cl_mode: str = "self-supervised"           # self-supervised 或 supervised
    cl_training_data_dir: str = "llm_training_data"
    cl_epochs: int = 10
    cl_batch_size: int = 64
    cl_temperature: float = 0.07
    cl_learning_rate: float = 1e-5
    cl_sample_rate: float = 1.0  # 超大数据集可以小一点
    force_retrain: bool = False
    cl_model_cache_dir: str = "finetuned_models"


def build_main_args():
    args = tyro.cli(MainArgs)

    # 把数据集对应的最优参数自动覆盖上去
    if args.use_dataset_config:
        try:
            from dataset_configs import get_dataset_config

            print(f"\n{'='*60}")
            print(f"加载 {args.data_name} 的数据集配置")
            print(f"{'='*60}")

            config = get_dataset_config(args.data_name)

            # 只覆盖那些用户没在命令行手动改过的参数
            # 判别方式：和 dataclass 默认值比较
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


            print(f"\n生效的 {args.data_name} 配置:")
            print(f"  - col_sim_threshold (γ) = {args.col_sim_threshold}")
            print(f"  - min_dis (m)           = {args.min_dis}")
            print(f"{'='*60}\n")

        except KeyError as e:
            print(f"\n[警告] {e}")
            print(f"[警告] 退回默认参数\n")
        except ImportError:
            print(f"\n[警告] dataset_configs.py 缺失，使用默认参数\n")

    # model_type -> 本地路径，setup.sh 已经把权重下到 model/ 下
    MODEL_MAPPING = {
        "minilm": "model/all-MiniLM-L12-v2",
        "modernbert": "model/modernbert",
        "gte-large": "/home/cjx/SAGEM/model/gte-large",
        "modernbert-server": "/home/cjx/SAGEM/model/modernbert"
    }

    # HF cache 风格目录的兼容：自动定位到 snapshots/<rev>
    # 仓库里 model/* 可能直接是 blobs/refs/snapshots 这种目录
    from pathlib import Path

    def _resolve_hf_snapshot_dir(p: str) -> str:
        try:
            base = Path(p)
            if not base.exists() or not base.is_dir():
                return p
            snaps = base / "snapshots"
            refs_main = base / "refs" / "main"
            if snaps.is_dir():
                # 优先用 refs/main 指的那个 revision
                if refs_main.is_file():
                    rev = refs_main.read_text().strip()
                    cand = snaps / rev
                    if cand.is_dir():
                        return str(cand)
                # 没有 refs/main 就取 snapshots 下任意一个目录
                subdirs = sorted([d for d in snaps.iterdir() if d.is_dir()])
                if subdirs:
                    return str(subdirs[-1])
            return p
        except Exception:
            return p

    # 用户给了 model_type 但没自己写路径时，按 mapping 套
    # 自己写了路径就以用户的为准
    defaults = MainArgs()
    user_specified_path = args.lm_model_or_path != defaults.lm_model_or_path
    if not user_specified_path and (args.model_type != defaults.model_type or args.lm_model_or_path == defaults.lm_model_or_path):
        if args.model_type in MODEL_MAPPING:
            args.lm_model_or_path = MODEL_MAPPING[args.model_type]
        else:
            print(f"\n[警告] 未知 model_type '{args.model_type}'")
            print(f"[警告] 可选: {list(MODEL_MAPPING.keys())}")
            print(f"[警告] 沿用默认: {args.lm_model_or_path}\n")

    # 老路径兼容：早期是放在 models/ 下，现在统一 model/
    # 脚本里如果还在传 models/...，且本地不存在，就回退到 model/
    from pathlib import Path as _Path

    legacy_map = {
        "models/modernbert-embed-base": "model/modernbert",
        "models/all-MiniLM-L12-v2": "model/all-MiniLM-L12-v2",
        "modernbert-embed-base": "model/modernbert",
        "all-MiniLM-L12-v2": "model/all-MiniLM-L12-v2",
        "gte-large": "/home/cjx/SAGEM/model/gte-large",
        "gte_large": "/home/cjx/SAGEM/model/gte-large",
        "modernbert-server": "/home/cjx/SAGEM/model/modernbert",
        "server-modernbert": "/home/cjx/SAGEM/model/modernbert",
        "/home/cjx/SAGEM/model/gte-large": "/home/cjx/SAGEM/model/gte-large",
        "/home/cjx/SAGEM/model/modernbert": "/home/cjx/SAGEM/model/modernbert",
    }
    if isinstance(args.lm_model_or_path, str):
        p = args.lm_model_or_path
        if p in legacy_map and (not _Path(p).exists() or p.startswith("/home/cjx/SAGEM/model/")):
            args.lm_model_or_path = legacy_map[p]

    # HF cache 风格的目录，再解析一次
    args.lm_model_or_path = _resolve_hf_snapshot_dir(args.lm_model_or_path)

    # 把最终生效的本地路径打出来，免得用户以为走了线上
    print(f"\n模型选择: {args.model_type}")
    print(f"   -> 路径: {args.lm_model_or_path}\n")

    # 按模型类型套推荐的 max_seq_length / batch_size
    # 用户显式指定过的不动
    MODEL_PARAMS = {
        "minilm": {
            "max_seq_length": 64,
            "batch_size": 512,
        },
        "modernbert": {
            "max_seq_length": 256,   # ModernBERT 支持很长（最大 8192）
            "batch_size": 256,       # 序列变长就得收一下 batch
        },
        "modernbert-server": {
            "max_seq_length": 256,
            "batch_size": 256,
        }
    }

    if args.model_type in MODEL_PARAMS:
        recommended_params = MODEL_PARAMS[args.model_type]

        params_changed = []
        if args.max_seq_length == defaults.max_seq_length:
            old_val = args.max_seq_length
            args.max_seq_length = recommended_params["max_seq_length"]
            params_changed.append(f"max_seq_length: {old_val} -> {args.max_seq_length}")

        if args.batch_size == defaults.batch_size:
            old_val = args.batch_size
            args.batch_size = recommended_params["batch_size"]
            params_changed.append(f"batch_size: {old_val} -> {args.batch_size}")

        if params_changed:
            print(f"按 {args.model_type} 自动调整:")
            for change in params_changed:
                print(f"   - {change}")
            print()

    return args
