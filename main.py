"""
主流程入口：跑一次完整的实体匹配流水线。

阶段顺序：参数解析 → 属性选择（带缓存）→ 重读选中列 → SentenceTransformer 编码 →
（可选）SmartTablePairing 配对 → 分层两两合并 → 计算 P/R/F1 + 错误分组 → 写
results/。每阶段都用 ResourceMonitor 采样 wall-clock / RAM / VRAM 峰值。

依赖入口：args.build_main_args、data.read_all_tables、selector.auto_selection、
merger.merge*、metrics.evaluate_log_with_output、result_logger.ResultLogger。
"""

from typing import List
from itertools import chain
from pathlib import Path
import sys
import os

# 把 src/ 下的子目录塞进 sys.path，老的 import 不用动
_base_path = Path(__file__).resolve().parent
sys.path.append(str(_base_path / 'src'))
for _sub in ['core', 'llm', 'data_chuli', 'training', 'utils']:
    sys.path.append(str(_base_path / 'src' / _sub))

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 30/40 系卡上开 TF32，能稳吃一波速度
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


def _wrap_data_parallel(model: SentenceTransformer, enabled: bool = False) -> bool:
    """需要 --multi-gpu 显式打开，否则不动"""
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        transformer = model._first_module()
        if not isinstance(transformer.auto_model, torch.nn.DataParallel):
            original_model = transformer.auto_model
            transformer.auto_model = torch.nn.DataParallel(original_model)
            # 这里得手动把 config 透出来，不然 ST 内部访问会炸
            transformer.auto_model.config = original_model.config
            print(f"  多GPU: DataParallel ({torch.cuda.device_count()} GPUs)")
            return True
    return False


def _unwrap_data_parallel(model: SentenceTransformer):
    transformer = model._first_module()
    if isinstance(transformer.auto_model, torch.nn.DataParallel):
        transformer.auto_model = transformer.auto_model.module

from args import build_main_args
from data import Table, read_all_tables, textify_table, read_ground_truth
from log import init_logger, log_args, log_time, log
from timer import Timer
from metrics import evaluate_log, evaluate_log_with_output
from selector import auto_selection
from selector_cache import get_cache_manager
from merger import merge, merge_parallel, merge_with_smart_pairing, merge_parallel_with_smart_pairing
from resource_monitor import ResourceMonitor, ensure_stage_metrics, format_bytes, update_stage_metrics
from result_logger import ResultLogger



if __name__ == '__main__':

    args = build_main_args()
    file_name = f"main"
    log_file_name = init_logger(file_name)
    log_args(args)
    log(log_file_name)

    # 初始化 result_logger
    result_logger = ResultLogger(dataset_name=args.data_name)
    result_logger.set_parameters(args)
    ensure_stage_metrics(args)
    # 挂到 args 上方便其他地方拿
    args.result_logger = result_logger
    run_monitor = ResourceMonitor()
    run_monitor.start()
    log(f"结果将保存到 results/ 目录")

    data_path = Path(args.data_path)
    full_data_path = data_path / args.data_name
    timer = Timer()

    # 属性选择（带缓存）
    cache_manager = get_cache_manager()

    # 先全量读一遍，给属性选择用
    T, tables_df = read_all_tables(full_data_path)

    manual_attrs = []
    if getattr(args, "manual_selected_attrs", ""):
        manual_attrs = [
            attr.strip() for attr in str(args.manual_selected_attrs).split(",")
            if attr.strip()
        ]

    if not args.eer_flag and manual_attrs:
        log("注意: 因为 eer_flag=False，--manual-selected-attrs 不会生效")

    if args.eer_flag:
        if manual_attrs:
            available_attrs = list(T[0].columns) if T else []
            invalid_attrs = [attr for attr in manual_attrs if attr not in available_attrs]
            if invalid_attrs:
                raise ValueError(
                    f"Manual selected attrs contain invalid columns: {invalid_attrs}. "
                    f"Available columns: {available_attrs}"
                )

            selected_attrs = ["tid"] + [attr for attr in manual_attrs if attr != "tid"]
            attribute_scores = {}
            log(f"使用手工指定的属性，跳过自动选择: {selected_attrs}")
            result_logger.log_attribute_selection(
                selected_attrs=selected_attrs,
                attribute_scores=attribute_scores,
                time=0.0,
                from_cache=False
            )
        else:
            # 不同模型缓存隔开
            cache = cache_manager.load_cache(
                args.data_name,
                args.col_sim_threshold,
                args.selection_rate,
                args.lm_model_or_path
            )

            if cache is not None:
                selected_attrs = cache.selected_attrs
                attribute_scores = {}  # 缓存里没存分数
                log("命中缓存，跳过属性选择")
                result_logger.log_attribute_selection(
                    selected_attrs=selected_attrs,
                    attribute_scores=attribute_scores,
                    time=0.0,
                    from_cache=True
                )
            else:
                selection_monitor = ResourceMonitor()
                selection_monitor.start()
                log("开始属性选择...")
                timer.start()
                selected_attrs, attribute_scores = auto_selection(tables_df, args)
                tm = timer.stop()
                selection_usage = selection_monitor.stop()
                stage_metrics = update_stage_metrics(args, "attribute_selection", selection_usage)
                log_time("selecting", tm)
                log(
                    f"Attribute selection resource usage | time={selection_usage.elapsed_time:.4f}s | "
                    f"peak_ram={format_bytes(selection_usage.peak_memory_bytes)}"
                    f"{'' if selection_usage.available else ' (psutil unavailable)'} | "
                    f"peak_vram={format_bytes(selection_usage.peak_gpu_memory_bytes)}"
                    f"{'' if selection_usage.gpu_available else ' (cuda unavailable)'} | "
                    f"stage_peak_ram={format_bytes(stage_metrics['peak_memory_bytes'])} | "
                    f"stage_peak_vram={format_bytes(stage_metrics['peak_gpu_memory_bytes'])}"
                )

                result_logger.log_attribute_selection(
                    selected_attrs=selected_attrs,
                    attribute_scores=attribute_scores,
                    time=tm,
                    from_cache=False
                )

                cache_manager.save_cache(
                    dataset_name=args.data_name,
                    selected_attrs=selected_attrs,
                    col_sim_threshold=args.col_sim_threshold,
                    selection_rate=args.selection_rate,
                    model_name=args.lm_model_or_path,
                    max_seq_length=args.max_seq_length,
                    selection_time=tm
                )
    else:
        selected_attrs = None
        attribute_scores = {}
        result_logger.log_attribute_selection(
            selected_attrs=[],
            attribute_scores={},
            time=0.0,
            from_cache=False
        )

# 读表 + 编码
    timer.start()
    T, tables_df = read_all_tables(
        full_data_path, selected_attrs=selected_attrs, sample_rate=args.data_sample_rate)
    tm = timer.stop()
    log_time("read all tables", tm)
    table_lens = [len(table) for table in tables_df]
    n = sum(table_lens)

    result_logger.log_data_loading(
        num_tables=len(tables_df),
        table_sizes=table_lens,
        time=tm,
        tables_df=tables_df,
        sample_size=5
    )

    table_ids = [table["tid"].tolist() for table in tables_df]
    tables = [Table(str(idx), table_id, list(range(len(table_id))))
              for idx, table_id in enumerate(table_ids)]

    # 表 -> 文本
    timer.start()
    table_sentences = [
        textify_table(table)
        for table in tables_df
    ]
    trust_code = "modernbert" in str(args.lm_model_or_path).lower()
    log(f"实例化 SentenceTransformer ...")
    model = SentenceTransformer(args.lm_model_or_path, trust_remote_code=trust_code, local_files_only=True)
    log(f"SentenceTransformer 实例化完成")
    model.max_seq_length = args.max_seq_length
    log(f"加载到设备: {args.device} (首次走 CUDA 会比较慢)")
    model.to(args.device)
    log(f"模型已加载到 {args.device}")
    _wrap_data_parallel(model, enabled=args.multi_gpu)
    table_embeddings = [
        model.encode(sentences, show_progress_bar=True,
                     batch_size=args.batch_size, normalize_embeddings=True)
        for sentences in table_sentences]
    all_embeddings = list(chain(*table_embeddings))
    all_embeddings = np.array(all_embeddings)
    tm = timer.stop()
    log_time("encode all tables", tm)

    result_logger.log_encoding(
        embedding_shape=list(all_embeddings.shape),
        model_name=args.lm_model_or_path,
        time=tm,
        all_embeddings=all_embeddings,
        table_sentences=table_sentences,
        sample_size=5
    )

    # tid -> 文本，evaluate 输出时用得上
    all_sentences_flat = list(chain(*table_sentences))
    tid_to_name = {tid: text for tid, text in enumerate(all_sentences_flat)}
    log(f"tid_to_name 共 {len(tid_to_name)} 条")

    # 这里没启用 CL，直接登一个 disabled
    result_logger.log_contrastive_learning(enabled=False)
    ground_truth = read_ground_truth(full_data_path)

    timer.start()

    # 合并阶段
    if args.use_smart_pairing:
        log("Using smart table pairing")
        if args.run_in_parallel:
            table = merge_parallel_with_smart_pairing(tables, all_embeddings, args)
        else:
            table = merge_with_smart_pairing(tables, all_embeddings, args)
    else:
        # 老路子：随机配对
        log("Using random table pairing")
        if args.run_in_parallel:
            table = merge_parallel(tables, all_embeddings, args)
        else:
            table = merge(tables, all_embeddings, args)
    tm = timer.stop()
    log_time("merging", tm)
    prediction = table.get_tuples()

    resource_usage = getattr(args, "_stage_resource_metrics", {})
    if resource_usage:
        log("=== Stage Resource Summary ===")
        stage_name_map = {
            "attribute_selection": "Attribute Selection",
            "table_pairing": "Table Pairing",
            "hierarchical_merging": "Hierarchical Merging",
        }
        for stage_key in ["attribute_selection", "table_pairing", "hierarchical_merging"]:
            stage_info = resource_usage.get(stage_key)
            if not stage_info:
                continue
            result_logger.log_resource_usage(stage_key, stage_info)
            log(
                f"{stage_name_map.get(stage_key, stage_key)} | "
                f"total_time={stage_info['total_time']:.4f}s | "
                f"peak_ram={format_bytes(stage_info['peak_memory_bytes'])} | "
                f"peak_vram={format_bytes(stage_info['peak_gpu_memory_bytes'])} | "
                f"invocations={stage_info['invocations']}"
                f"{'' if stage_info.get('available', False) else ' (psutil unavailable)'}"
            )

    # 记录合并阶段总结
    result_logger.log_merging_summary(
        total_time=tm,
        final_tuples=len(prediction)
    )

    # 把合并后的表内容落盘，方便人工核对
    from datetime import datetime
    merge_table_file = f"logs/merge_table_content_{args.data_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(merge_table_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"合并后表内容 (用于评分)\n")
        f.write(f"数据集: {args.data_name}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总元组数: {len(prediction)}\n")
        f.write(f"总实体数: {sum(len(t) for t in prediction)}\n")
        f.write(f"{'='*80}\n\n")

        for idx, tuple_tids in enumerate(prediction, 1):
            f.write(f"元组 #{idx} (含 {len(tuple_tids)} 个实体)\n")
            f.write(f"  TIDs: {tuple_tids}\n")
            f.write(f"  实体内容:\n")
            for i, tid in enumerate(tuple_tids, 1):
                text = tid_to_name.get(tid, f"Unknown-{tid}")
                f.write(f"    {i}. [{tid}] {text}\n")
            f.write("\n")

    log(f"合并后表内容已写入: {merge_table_file}")

    # 合并后评估
    result_logger.log_evaluation(ground_truth, prediction, is_final=False)
    merge_output_file = f"logs/entity_groups_after_merge_{args.data_name}.txt"
    evaluate_log_with_output(ground_truth, prediction, tid_to_name, merge_output_file)

    # 终评估（这版没有剪枝阶段，直接复用同一份 prediction）
    result_logger.log_evaluation(ground_truth, prediction, is_final=True)
    final_output_file = f"logs/entity_groups_final_{args.data_name}.txt"
    evaluate_log_with_output(ground_truth, prediction, tid_to_name, final_output_file)

    run_usage = run_monitor.stop()
    phase_times = {
        "attribute_selection": float(result_logger.attribute_selection.time),
        "data_loading": float(result_logger.data_loading.time),
        "encoding": float(result_logger.encoding.time),
        "contrastive_learning": float(result_logger.contrastive_learning.time),
        "merging": float(result_logger.merging.total_time),
    }
    run_summary = {
        "phase_times": phase_times,
        "total_time": float(run_usage.elapsed_time),
        "peak_memory_bytes": int(run_usage.peak_memory_bytes),
        "peak_memory_human": format_bytes(run_usage.peak_memory_bytes),
        "peak_gpu_memory_bytes": int(run_usage.peak_gpu_memory_bytes),
        "peak_gpu_memory_human": format_bytes(run_usage.peak_gpu_memory_bytes),
        "memory_available": bool(run_usage.available),
        "gpu_available": bool(run_usage.gpu_available),
    }
    result_logger.log_run_summary(run_summary)

    # 落盘
    json_file, txt_file = result_logger.save_results()
    log(f"中间结果已写入:")
    log(f"  JSON: {json_file}")
    log(f"  TXT : {txt_file}")

    log("=== Run Summary ===")
    for phase_name, phase_time in phase_times.items():
        log(f"{phase_name}: {phase_time:.4f}s")
    log(f"total_time: {run_usage.elapsed_time:.4f}s")
    log(
        f"max_ram: {format_bytes(run_usage.peak_memory_bytes)}"
        f"{'' if run_usage.available else ' (psutil unavailable)'}"
    )
    log(
        f"max_vram: {format_bytes(run_usage.peak_gpu_memory_bytes)}"
        f"{'' if run_usage.gpu_available else ' (cuda unavailable)'}"
    )
