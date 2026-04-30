"""CLI 入口：编排 采样 → 分析 → 生成 → 格式化 全流程。"""

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
    _D = GeneratorConfig()  # 默认值唯一来源
    parser = argparse.ArgumentParser(
        description="LLM 驱动的实体匹配训练数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Ollama 后端
  python -m llm_data_generator.main --dataset Geo --data-dir data --num-entities 200

  # vLLM 后端
  python -m llm_data_generator.main --backend vllm --api-url http://localhost:8000 \\
      --model /path/to/Qwen3.5-27B --dataset Geo --data-dir data

  # 只做分析
  python -m llm_data_generator.main --dataset Geo --data-dir data --analyze-only

  # 所有数据集
  python -m llm_data_generator.main --dataset all --data-dir data
        """,
    )
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_REGISTRY.keys()) + ["all"],
                        help="数据集名称或 'all'")
    parser.add_argument("--data-dir", default="data",
                        help="数据目录路径 (默认: data)")
    parser.add_argument("--output-dir", default="llm_training_data",
                        help="输出目录 (默认: llm_training_data)")
    parser.add_argument("--backend", default=_D.backend,
                        choices=["ollama", "vllm", "api"],
                        help=f"LLM 后端 (默认: {_D.backend})")
    parser.add_argument("--api-url", default=_D.api_url,
                        help=f"LLM API 地址 (默认: {_D.api_url})")
    parser.add_argument("--api-key", default=_D.api_key,
                        help="External OpenAI-compatible API key; falls back to OPENAI_API_KEY or LLM_API_KEY")
    parser.add_argument("--model", default=_D.model,
                        help=f"模型名称 (默认: {_D.model})")
    parser.add_argument("--temperature", type=float, default=_D.temperature,
                        help=f"LLM 生成温度 (默认: {_D.temperature})")
    parser.add_argument("--analysis-temperature", type=float, default=_D.analysis_temperature,
                        help=f"LLM 分析温度 (默认: {_D.analysis_temperature})")
    parser.add_argument("--num-entities", type=int, default=_D.num_entities,
                        help=f"目标生成 entity group 数 (默认: {_D.num_entities})")
    parser.add_argument("--batch-size", type=int, default=_D.batch_size,
                        help=f"每批 LLM 调用生成数 (默认: {_D.batch_size})")
    parser.add_argument("--sample-size", type=int, default=_D.sample_size,
                        help=f"每表分析采样数 (默认: {_D.sample_size})")
    parser.add_argument("--seed", type=int, default=_D.seed,
                        help=f"随机种子 (默认: {_D.seed})")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只做分析不生成")
    # 分析缓存复用开关(互斥两个别名,任何一个打开都等价)
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-analysis", dest="reuse_analysis", action="store_true",
        default=None,
        help="复用已有的分析提示词缓存(analysis_cache_<model>.json),跳过 Stage 2。"
             "缓存不存在时仍会执行分析。默认关闭:每次都重新分析。",
    )
    reuse_group.add_argument(
        "--skip-analysis", dest="reuse_analysis", action="store_true",
        default=None,
        help="同 --reuse-analysis,保留的旧参数名。",
    )
    reuse_group.add_argument(
        "--force-analysis", dest="reuse_analysis", action="store_false",
        help="强制重新分析(显式关闭缓存复用,即使缓存存在也会重新调 LLM)。",
    )
    parser.add_argument("--max-retries", type=int, default=_D.max_retries,
                        help=f"LLM 调用最大重试次数 (默认: {_D.max_retries})")
    parser.add_argument("--timeout", type=int, default=_D.timeout,
                        help=f"LLM 请求超时秒数 (默认: {_D.timeout})")
    parser.add_argument("--num-ctx", type=int, default=_D.num_ctx,
                        help=f"上下文窗口大小 (默认: {_D.num_ctx})")
    parser.add_argument("--max-workers", type=int, default=_D.max_workers,
                        help=f"并发生成线程数 (默认: {_D.max_workers})")
    return parser.parse_args()


def _create_client(gen_config: GeneratorConfig, token_tracker=None):
    """根据 backend 创建对应的 LLM 客户端。"""
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
    """处理单个数据集的完整流程。

    Args:
        reuse_analysis: 若为 True 且 analysis_cache 已存在,则跳过 Stage 2
                        直接复用缓存的分析提示词;缓存不存在时仍会执行分析。
                        默认 False,每次重新分析。
        global_tracker: 全局 TokenTracker(跨数据集累计),可选
    """
    print(f"\n{'=' * 60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'=' * 60}")

    # 1. 加载配置
    config = get_dataset_config(dataset_name)
    data_dir = Path(data_dir_root) / dataset_name
    output_dir = Path(output_dir_root) / dataset_name

    if not data_dir.exists():
        print(f"  错误: 数据目录不存在 {data_dir}")
        return

    # 2. 加载数据
    print("\nStage 1: 数据采样...")
    tables = load_tables(str(data_dir))

    # 如果配置不存在，自动检测
    if config is None:
        columns = [c for c in tables[0].columns if c != "tid"]
        config = auto_detect_config(dataset_name, columns)
        print(f"  自动检测配置: 列={columns}")

    # 采样
    sampled = sample_records(tables, gen_config.sample_size, gen_config.seed)

    # 3. 初始化 LLM 客户端（使用全局 tracker 或创建独立 tracker）
    tracker = global_tracker if global_tracker is not None else TokenTracker(model=gen_config.model)
    client, backend_label = _create_client(gen_config, token_tracker=tracker)

    if not client.check_connection():
        print(f"\n  错误: 无法连接到 {backend_label} ({gen_config.api_url})")
        return

    available_models = client.list_models()
    if available_models and gen_config.model not in available_models:
        print(f"  警告: 模型 '{gen_config.model}' 可能未安装")
        print(f"  可用模型: {', '.join(available_models)}")

    # 模型名清理（用于文件名）
    model_tag = gen_config.model.replace("/", "-").replace(":", "-")

    # 4. 分析提示词 —— 四种情况分别处理并明确告知
    analysis_cache = output_dir / f"analysis_cache_{model_tag}.json"
    cache_exists = analysis_cache.exists()

    if reuse_analysis and cache_exists:
        # ① 开关打开 + 缓存存在 → 直接复用
        print(f"\nStage 2: [复用缓存] 跳过分析,加载已有提示词")
        print(f"  ← {analysis_cache}")
        print(f"  提示:如需重新生成提示词,去掉 --reuse-analysis 或加 --force-analysis")
        analysis = load_analysis(str(analysis_cache))
    elif reuse_analysis and not cache_exists:
        # ② 开关打开 + 缓存不存在 → 仍需执行分析
        print(f"\nStage 2: [缓存未命中] --reuse-analysis 已设置,但缓存不存在,将执行首次分析")
        print(f"  缓存将写入 {analysis_cache}")
        client._current_stage = "analysis"
        analysis = analyze_dataset(
            client, config, sampled, len(tables), gen_config
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(analysis, str(analysis_cache))
    elif not reuse_analysis and cache_exists:
        # ③ 开关关闭 + 缓存存在 → 重新分析并覆盖缓存
        print(f"\nStage 2: [重新分析] 缓存存在但未启用复用,将覆盖 {analysis_cache}")
        print(f"  提示:若希望直接使用已有提示词,改用 --reuse-analysis")
        client._current_stage = "analysis"
        analysis = analyze_dataset(
            client, config, sampled, len(tables), gen_config
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(analysis, str(analysis_cache))
    else:
        # ④ 开关关闭 + 缓存不存在 → 首次分析
        print(f"\nStage 2: [首次分析] 缓存不存在,执行完整分析流程")
        client._current_stage = "analysis"
        analysis = analyze_dataset(
            client, config, sampled, len(tables), gen_config
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(analysis, str(analysis_cache))

    if analyze_only:
        print("\n分析结果:")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return

    # 5. 生成
    client._current_stage = "generation"
    entity_groups = generate_entity_groups(
        client, config, analysis,
        gen_config.num_entities, gen_config.batch_size,
        max_workers=gen_config.max_workers,
    )

    if not entity_groups:
        print("  错误: 未能生成任何 entity groups")
        return

    # 6. 格式化并保存
    print("\nStage 4: 构建正样本对并保存...")
    pairs, metadata = build_pairs(
        entity_groups, config, gen_config.seed,
    )

    output_path = output_dir / f"{dataset_name}_{model_tag}.json"
    save_output(entity_groups, pairs, metadata, str(output_path))

    # 输出摘要
    print(f"\n{'─' * 40}")
    print(f"完成: {dataset_name}")
    print(f"  Entity groups: {len(entity_groups)}")
    print(f"  正样本对: {len(pairs)}")
    print(f"  输出文件: {output_path}")
    print(f"{'─' * 40}")


def main():
    args = parse_args()

    # 从 CLI 参数构建统一配置
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

    # 解析缓存复用开关:CLI 未显式指定时用 config 默认值
    reuse_analysis = (
        args.reuse_analysis
        if args.reuse_analysis is not None
        else gen_config.reuse_analysis_cache
    )
    gen_config.reuse_analysis_cache = reuse_analysis  # 回写便于下游读取

    print("=" * 60)
    print("LLM 训练数据生成器")
    print(f"  后端: {gen_config.backend}")
    print(f"  模型: {gen_config.model}")
    print(f"  API:  {gen_config.api_url}")
    print(f"  目标: {gen_config.num_entities} entity groups")
    print(f"  分析缓存: {'复用(若存在)' if reuse_analysis else '每次重新生成'}")
    print("=" * 60)

    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]

    # 全局 Token 统计器 — 跨所有数据集累计
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
            print(f"\n  处理 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 全流程结束：输出总用量并保存
    global_tracker.report()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = gen_config.model.replace("/", "-").replace(":", "-")
    global_tracker.save(str(output_dir / f"total_token_stats_{model_tag}.json"))

    print("全部完成！")


if __name__ == "__main__":
    main()
