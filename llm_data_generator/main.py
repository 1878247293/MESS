"""CLI entry point: chain sample -> analyze -> generate -> format together."""

import argparse
import json
import os
import sys
from pathlib import Path

from .config import get_dataset_config, auto_detect_config, GeneratorConfig, DATASET_REGISTRY
from .api_client import ApiClient
from .ollama_client import OllamaClient
from .vllm_client import VllmClient
from .sampler import (
    load_tables, sample_records,
)
from .analyzer import analyze_dataset, save_analysis, load_analysis
from .generator import generate_entity_groups
from .formatter import build_pairs, save_output
from .token_tracker import TokenTracker


def parse_args():
    _D = GeneratorConfig()  # single source of default values
    parser = argparse.ArgumentParser(
        description="LLM-driven entity matching training data generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ollama backend
  python -m llm_data_generator.main --dataset Geo --data-dir data --num-entities 200

  # vLLM backend
  python -m llm_data_generator.main --backend vllm --api-url http://localhost:8000 \\
      --model /path/to/Qwen3.5-27B --dataset Geo --data-dir data

  # analysis only
  python -m llm_data_generator.main --dataset Geo --data-dir data --analyze-only

  # all datasets
  python -m llm_data_generator.main --dataset all --data-dir data
        """,
    )
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_REGISTRY.keys()) + ["all"],
                        help="dataset name or 'all'")
    parser.add_argument("--data-dir", default="data",
                        help="data directory path (default: data)")
    parser.add_argument("--output-dir", default="llm_training_data",
                        help="output directory (default: llm_training_data)")
    parser.add_argument("--backend", default=_D.backend,
                        choices=["ollama", "vllm", "api"],
                        help=f"LLM backend (default: {_D.backend})")
    parser.add_argument("--api-url", default=_D.api_url,
                        help=f"LLM API address (default: {_D.api_url})")
    parser.add_argument("--api-key", default=_D.api_key,
                        help="External OpenAI-compatible API key; falls back to OPENAI_API_KEY or LLM_API_KEY")
    parser.add_argument("--model", default=_D.model,
                        help=f"model name (default: {_D.model})")
    parser.add_argument("--temperature", type=float, default=_D.temperature,
                        help=f"LLM generation temperature (default: {_D.temperature})")
    parser.add_argument("--analysis-temperature", type=float, default=_D.analysis_temperature,
                        help=f"LLM analysis temperature (default: {_D.analysis_temperature})")
    parser.add_argument("--num-entities", type=int, default=_D.num_entities,
                        help=f"target number of entity groups to generate (default: {_D.num_entities})")
    parser.add_argument("--batch-size", type=int, default=_D.batch_size,
                        help=f"number generated per LLM-call batch (default: {_D.batch_size})")
    parser.add_argument("--sample-size", type=int, default=_D.sample_size,
                        help=f"number of rows sampled per table for analysis (default: {_D.sample_size})")
    parser.add_argument("--seed", type=int, default=_D.seed,
                        help=f"random seed (default: {_D.seed})")
    parser.add_argument("--analyze-only", action="store_true",
                        help="analyze only, do not generate")
    # cache-reuse switch: two mutually exclusive aliases, either works
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-analysis", dest="reuse_analysis", action="store_true",
        default=None,
        help="If analysis_cache_<model>.json already exists, reuse it directly and skip Stage 2. "
             "Analysis is still performed when the cache does not exist. By default, re-analyze every time.",
    )
    reuse_group.add_argument(
        "--skip-analysis", dest="reuse_analysis", action="store_true",
        default=None,
        help="Old name for --reuse-analysis; equivalent.",
    )
    reuse_group.add_argument(
        "--force-analysis", dest="reuse_analysis", action="store_false",
        help="Re-run once even if the cache exists.",
    )
    parser.add_argument("--max-retries", type=int, default=_D.max_retries,
                        help=f"maximum retries for LLM calls (default: {_D.max_retries})")
    parser.add_argument("--timeout", type=int, default=_D.timeout,
                        help=f"LLM request timeout in seconds (default: {_D.timeout})")
    parser.add_argument("--num-ctx", type=int, default=_D.num_ctx,
                        help=f"context window size (default: {_D.num_ctx})")
    parser.add_argument("--max-workers", type=int, default=_D.max_workers,
                        help=f"number of concurrent generation threads (default: {_D.max_workers})")
    return parser.parse_args()


def _create_client(gen_config: GeneratorConfig, token_tracker=None):
    """Get the client corresponding to the backend"""
    client_kwargs = dict(
        base_url=gen_config.api_url,
        model=gen_config.model,
        temperature=gen_config.temperature,
        max_retries=gen_config.max_retries,
        timeout=gen_config.timeout,
        num_ctx=gen_config.num_ctx,
        token_tracker=token_tracker,
    )
    if gen_config.backend == "vllm":
        return VllmClient(**client_kwargs), "vLLM"
    if gen_config.backend == "api":
        api_key = gen_config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("backend=api requires --api-key or OPENAI_API_KEY / LLM_API_KEY")
        return ApiClient(api_key=api_key, **client_kwargs), "OpenAI-Compatible API"
    else:
        return OllamaClient(**client_kwargs), "Ollama"


def process_dataset(dataset_name: str, gen_config: GeneratorConfig,
                    data_dir_root: str, output_dir_root: str,
                    analyze_only: bool = False, reuse_analysis: bool = False,
                    global_tracker: TokenTracker = None):
    """Run the full pipeline for a single dataset

    Args:
        reuse_analysis: skip Stage 2 on a cache hit; still analyze on a miss
        global_tracker: global TokenTracker (accumulated across datasets)
    """
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # 1. config
    config = get_dataset_config(dataset_name)
    data_dir = Path(data_dir_root) / dataset_name
    output_dir = Path(output_dir_root) / dataset_name

    if not data_dir.exists():
        print(f"  data directory does not exist: {data_dir}")
        return

    # 2. data
    print("\nStage 1: sampling...")
    tables = load_tables(str(data_dir))

    # for unregistered datasets, build config from the column names
    if config is None:
        columns = [c for c in tables[0].columns if c != "tid"]
        config = auto_detect_config(dataset_name, columns)
        print(f"  auto-detected columns: {columns}")

    sampled = sample_records(tables, gen_config.sample_size, gen_config.seed)

    # 3. client (use the global tracker or start one of our own)
    tracker = global_tracker if global_tracker is not None else TokenTracker(model=gen_config.model)
    client, backend_label = _create_client(gen_config, token_tracker=tracker)

    if not client.check_connection():
        print(f"\n  cannot connect to {backend_label} ({gen_config.api_url})")
        return

    available_models = client.list_models()
    if available_models and gen_config.model not in available_models:
        print(f"  note: model '{gen_config.model}' is not in the available list")
        print(f"  available: {', '.join(available_models)}")

    # a model name with / or : cannot be used directly as a file name
    model_tag = gen_config.model.replace("/", "-").replace(":", "-")

    # 4. analysis stage: four combinations
    analysis_cache = output_dir / f"analysis_cache_{model_tag}.json"
    cache_exists = analysis_cache.exists()

    if reuse_analysis and cache_exists:
        # (1) switch on + cache present -> use directly
        print(f"\nStage 2: [reuse cache] skipping analysis")
        print(f"  <- {analysis_cache}")
        print(f"  to re-run, drop --reuse-analysis or add --force-analysis")
        analysis = load_analysis(str(analysis_cache))
    elif reuse_analysis and not cache_exists:
        # (2) switch on + no cache -> still run once
        print(f"\nStage 2: [cache miss] first analysis, the result will be written to {analysis_cache}")
        client._current_stage = "analysis"
        analysis = analyze_dataset(
            client, config, sampled, len(tables), gen_config
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(analysis, str(analysis_cache))
    elif not reuse_analysis and cache_exists:
        # (3) switch off + cache present -> re-run and overwrite
        print(f"\nStage 2: [re-run] overwriting {analysis_cache}")
        print(f"  to reuse directly, switch to --reuse-analysis")
        client._current_stage = "analysis"
        analysis = analyze_dataset(
            client, config, sampled, len(tables), gen_config
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(analysis, str(analysis_cache))
    else:
        # (4) switch off + no cache -> first analysis
        print(f"\nStage 2: [first analysis]")
        client._current_stage = "analysis"
        analysis = analyze_dataset(
            client, config, sampled, len(tables), gen_config
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(analysis, str(analysis_cache))

    if analyze_only:
        print("\nAnalysis result:")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return

    # 5. generation
    client._current_stage = "generation"
    entity_groups = generate_entity_groups(
        client, config, analysis,
        gen_config.num_entities, gen_config.batch_size,
        max_workers=gen_config.max_workers,
    )

    if not entity_groups:
        print("  no entity groups were generated")
        return

    # 6. build positive pairs and write to disk
    print("\nStage 4: building positive pairs and saving...")
    pairs, metadata = build_pairs(
        entity_groups, config, gen_config.seed,
    )

    output_path = output_dir / f"{dataset_name}_{model_tag}.json"
    save_output(entity_groups, pairs, metadata, str(output_path))

    print(f"\n{'-' * 40}")
    print(f"Done: {dataset_name}")
    print(f"  Entity groups: {len(entity_groups)}")
    print(f"  positive pairs: {len(pairs)}")
    print(f"  output: {output_path}")
    print(f"{'-' * 40}")


def main():
    args = parse_args()

    # CLI -> GeneratorConfig
    gen_config = GeneratorConfig(
        backend=args.backend,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        sample_size=args.sample_size,
        num_entities=args.num_entities,
        batch_size=args.batch_size,
        temperature=args.temperature,
        analysis_temperature=args.analysis_temperature,
        max_retries=args.max_retries,
        seed=args.seed,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
        max_workers=args.max_workers,
    )

    # use the config default when the CLI does not specify it explicitly
    reuse_analysis = (
        args.reuse_analysis
        if args.reuse_analysis is not None
        else gen_config.reuse_analysis_cache
    )
    gen_config.reuse_analysis_cache = reuse_analysis

    print("=" * 60)
    print("LLM training data generator")
    print(f"  backend: {gen_config.backend}")
    print(f"  model: {gen_config.model}")
    print(f"  API:  {gen_config.api_url}")
    print(f"  target: {gen_config.num_entities} entity groups")
    print(f"  analysis cache: {'reuse (if present)' if reuse_analysis else 're-run every time'}")
    print("=" * 60)

    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]

    # token accumulation across datasets
    global_tracker = TokenTracker(model=gen_config.model)

    for dataset_name in datasets:
        try:
            process_dataset(
                dataset_name, gen_config,
                args.data_dir, args.output_dir,
                args.analyze_only, reuse_analysis,
                global_tracker=global_tracker,
            )
        except Exception as e:
            print(f"\n  {dataset_name} error: {e}")
            import traceback
            traceback.print_exc()

    # total usage
    global_tracker.report()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = gen_config.model.replace("/", "-").replace(":", "-")
    global_tracker.save(str(output_dir / f"total_token_stats_{model_tag}.json"))

    print("done")


if __name__ == "__main__":
    main()
