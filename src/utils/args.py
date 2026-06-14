"""
Command-line argument definition and parsing (based on tyro).

`MainArgs` is a single dataclass covering all main-pipeline switches: data paths, attribute selection,
merging, parallelism, smart pairing, contrastive learning, model selection, candidate output, etc.
After parsing, `build_main_args()` also performs these automations:
- apply the best parameters from `dataset_configs.py` by `data_name` (values explicitly given by the user are left untouched);
- automatically map `model_type` -> local weight path;
- automatically resolve HF-cache-style directories to `snapshots/<rev>`;
- set default `max_seq_length` / `batch_size` by model type.
"""

import os
from dataclasses import dataclass

# these two variables must be set before importing HF / sentence-transformers
# otherwise there will be occasional network probes that are painfully slow
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import tyro


@dataclass
class MainArgs:
    data_path: str = "data"
    data_name: str = "Geo"  # default to Geo, since that dataset is already tuned

    # selecting
    eer_flag: bool = True
    col_sim_threshold: float = 0.8  # gamma; grid search found 0.8 best on Geo
    selection_rate: float = 0.2  # r
    manual_selected_attrs: str = ""  # manually specified attributes, comma-separated; when non-empty, skip auto selection
    data_sample_rate: float = 1.0  # large datasets can be downsampled
    # merging
    k: int = 1
    min_dis: float = 0.5  # m, the distance threshold
    # parallel
    run_in_parallel: bool = False
    multi_gpu: bool = False  # multi-GPU DataParallel

    # smart table pairing
    use_smart_pairing: bool = False

    # model
    # minilm or modernbert; the path defaults to a local one
    model_type: str = "modernbert"
    lm_model_or_path: str = "model/modernbert"  # offline path; change to an HF repo to go online
    device: str = "cuda"
    seed: int = 3407
    max_seq_length: int = 64
    batch_size: int = 512


    # whether to automatically apply the best parameters for each dataset
    use_dataset_config: bool = True

    # candidate set
    output_candidates: bool = True
    candidates_output_dir: str = "candidates_output"

    # contrastive learning
    use_contrastive_learning: bool = False
    cl_training_data_dir: str = "llm_training_data"
    cl_epochs: int = 10
    cl_batch_size: int = 64
    cl_temperature: float = 0.07
    cl_learning_rate: float = 1e-5
    cl_sample_rate: float = 1.0  # can be smaller for very large datasets
    force_retrain: bool = False
    cl_model_cache_dir: str = "finetuned_models"


def build_main_args():
    args = tyro.cli(MainArgs)

    # automatically override with the best parameters for the dataset
    if args.use_dataset_config:
        try:
            from dataset_configs import get_dataset_config

            print(f"\n{'='*60}")
            print(f"Loading dataset config for {args.data_name}")
            print(f"{'='*60}")

            config = get_dataset_config(args.data_name)

            # only override parameters the user did not manually change on the command line
            # detection method: compare against the dataclass default values
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


            print(f"\nEffective {args.data_name} config:")
            print(f"  - col_sim_threshold (γ) = {args.col_sim_threshold}")
            print(f"  - min_dis (m)           = {args.min_dis}")
            print(f"{'='*60}\n")

        except KeyError as e:
            print(f"\n[Warning] {e}")
            print(f"[Warning] falling back to default parameters\n")
        except ImportError:
            print(f"\n[Warning] dataset_configs.py is missing, using default parameters\n")

    # model_type -> local path; setup.sh has already downloaded the weights into model/
    MODEL_MAPPING = {
        "minilm": "model/all-MiniLM-L12-v2",
        "modernbert": "model/modernbert",
        "gte-large": "/home/cjx/SAGEM/model/gte-large",
        "modernbert-server": "/home/cjx/SAGEM/model/modernbert"
    }

    # compatibility with HF-cache-style directories: automatically locate snapshots/<rev>
    # model/* in the repo may directly be a blobs/refs/snapshots-style directory
    from pathlib import Path

    def _resolve_hf_snapshot_dir(p: str) -> str:
        try:
            base = Path(p)
            if not base.exists() or not base.is_dir():
                return p
            snaps = base / "snapshots"
            refs_main = base / "refs" / "main"
            if snaps.is_dir():
                # prefer the revision pointed to by refs/main
                if refs_main.is_file():
                    rev = refs_main.read_text().strip()
                    cand = snaps / rev
                    if cand.is_dir():
                        return str(cand)
                # if there is no refs/main, take any directory under snapshots
                subdirs = sorted([d for d in snaps.iterdir() if d.is_dir()])
                if subdirs:
                    return str(subdirs[-1])
            return p
        except Exception:
            return p

    # when the user gives model_type but does not write a path, apply the mapping
    # if a path was written explicitly, the user's value takes precedence
    defaults = MainArgs()
    user_specified_path = args.lm_model_or_path != defaults.lm_model_or_path
    if not user_specified_path and (args.model_type != defaults.model_type or args.lm_model_or_path == defaults.lm_model_or_path):
        if args.model_type in MODEL_MAPPING:
            args.lm_model_or_path = MODEL_MAPPING[args.model_type]
        else:
            print(f"\n[Warning] unknown model_type '{args.model_type}'")
            print(f"[Warning] options: {list(MODEL_MAPPING.keys())}")
            print(f"[Warning] keeping default: {args.lm_model_or_path}\n")

    # legacy path compatibility: early on these were under models/, now unified under model/
    # if a script still passes models/... and it does not exist locally, fall back to model/
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

    # resolve HF-cache-style directories once more
    args.lm_model_or_path = _resolve_hf_snapshot_dir(args.lm_model_or_path)

    # print the final effective local path, so the user does not think it went online
    print(f"\nModel selection: {args.model_type}")
    print(f"   -> path: {args.lm_model_or_path}\n")

    # apply the recommended max_seq_length / batch_size by model type
    # values explicitly specified by the user are left untouched
    MODEL_PARAMS = {
        "minilm": {
            "max_seq_length": 64,
            "batch_size": 512,
        },
        "modernbert": {
            "max_seq_length": 256,   # ModernBERT supports very long sequences (up to 8192)
            "batch_size": 256,       # longer sequences require a smaller batch
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
            print(f"Auto-adjusted for {args.model_type}:")
            for change in params_changed:
                print(f"   - {change}")
            print()

    return args
