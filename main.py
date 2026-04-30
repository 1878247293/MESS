from typing import List
from itertools import chain
from pathlib import Path
import sys
import os

# 自动将 src 及其子目录添加到搜索路径，以保持旧的 import 语句可用
_base_path = Path(__file__).resolve().parent
sys.path.append(str(_base_path / 'src'))
for _sub in ['core', 'llm', 'data_chuli', 'training', 'utils']:
    sys.path.append(str(_base_path / 'src' / _sub))

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# 性能优化：启用 TensorFloat32 加速 (针对 RTX 30/40 系列显卡)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')


def _wrap_data_parallel(model: SentenceTransformer, enabled: bool = False) -> bool:
    """多 GPU 时用 DataParallel 包裹底层 transformer，需 --multi-gpu 开启"""
    if enabled and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        transformer = model._first_module()
        if not isinstance(transformer.auto_model, torch.nn.DataParallel):
            original_model = transformer.auto_model
            transformer.auto_model = torch.nn.DataParallel(original_model)
            # 暴露 config 等属性，避免 sentence-transformers 内部访问报错
            transformer.auto_model.config = original_model.config
            print(f"  多GPU: DataParallel ({torch.cuda.device_count()} GPUs)")
            return True
    return False


def _unwrap_data_parallel(model: SentenceTransformer):
    """解除 DataParallel 包裹"""
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

# ========== PathCL-EM: 导入新模块 ==========



if __name__ == '__main__':

    args = build_main_args()#参数解析
    file_name = f"main"
    log_file_name = init_logger(file_name)
    log_args(args)
    log(log_file_name)

    # ========== 初始化结果记录器 ==========
    result_logger = ResultLogger(dataset_name=args.data_name)
    result_logger.set_parameters(args)
    ensure_stage_metrics(args)
    # 将 result_logger 附加到 args 上，以便在其他模块中使用
    args.result_logger = result_logger
    run_monitor = ResourceMonitor()
    run_monitor.start()
    log(f"结果将保存到 results/ 目录")
    # ========== 初始化结果记录器结束 ==========

    data_path = Path(args.data_path)#数据路径
    full_data_path = data_path / args.data_name
    timer = Timer()#计时器

    # ========== 属性选择(支持缓存) ==========
    cache_manager = get_cache_manager()#缓存管理器

    # pd.df - 首次读取(用于属性选择)
    T, tables_df = read_all_tables(full_data_path)

    manual_attrs = []
    if getattr(args, "manual_selected_attrs", ""):
        manual_attrs = [
            attr.strip() for attr in str(args.manual_selected_attrs).split(",")
            if attr.strip()
        ]

    if not args.eer_flag and manual_attrs:
        log("⚠️  --manual-selected-attrs 被忽略，因为 eer_flag=False（属性选择阶段未启用）")

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
            log(f"🛠️ 使用手工指定属性，跳过自动属性选择: {selected_attrs}")
            result_logger.log_attribute_selection(
                selected_attrs=selected_attrs,
                attribute_scores=attribute_scores,
                time=0.0,
                from_cache=False
            )
        else:
            # 尝试从缓存加载（不同模型独立缓存）
            cache = cache_manager.load_cache(
                args.data_name,
                args.col_sim_threshold,
                args.selection_rate,
                args.lm_model_or_path
            )

            if cache is not None:
                # 使用缓存的结果
                selected_attrs = cache.selected_attrs
                attribute_scores = {}  # 缓存中没有保存属性分数，使用空字典
                log("⚡ 跳过属性选择阶段(使用缓存)")
                # 记录到 result_logger（使用缓存）
                result_logger.log_attribute_selection(
                    selected_attrs=selected_attrs,
                    attribute_scores=attribute_scores,
                    time=0.0,
                    from_cache=True
                )
            else:
                selection_monitor = ResourceMonitor()
                selection_monitor.start()
                # 执行属性选择并保存缓存
                log("🔍 开始属性选择...")
                timer.start()
                selected_attrs, attribute_scores = auto_selection(tables_df, args)#执行属性选择
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

                # 记录到 result_logger
                result_logger.log_attribute_selection(
                    selected_attrs=selected_attrs,
                    attribute_scores=attribute_scores,
                    time=tm,
                    from_cache=False
                )

                # 保存到缓存（两个模型共用同一缓存）
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
        # 记录到 result_logger（未启用EER）
        result_logger.log_attribute_selection(
            selected_attrs=[],
            attribute_scores={},
            time=0.0,
            from_cache=False
        )
    # ========== 属性选择结束 ==========
    
# ==========读表和编码 =========
    timer.start()
    T, tables_df = read_all_tables(#返回表数量 T 和 DataFrame 列表 tables_df。
        full_data_path, selected_attrs=selected_attrs, sample_rate=args.data_sample_rate)
    tm = timer.stop()
    log_time("read all tables", tm)
    table_lens = [len(table) for table in tables_df]#每个表的行数
    n = sum(table_lens)

    # 记录数据加载结果
    result_logger.log_data_loading(
        num_tables=len(tables_df),
        table_sizes=table_lens,
        time=tm,
        tables_df=tables_df,  # 传入原始表数据
        sample_size=5
    )

    table_ids = [table["tid"].tolist() for table in tables_df]#每个表的 tid 列表
    # data.Table
    tables = [Table(str(idx), table_id, list(range(len(table_id))))
              for idx, table_id in enumerate(table_ids)]

    # ========== 文本化表格 ==========
    timer.start()
    table_sentences = [
        textify_table(table)
        for table in tables_df
    ]
    trust_code = "modernbert" in str(args.lm_model_or_path).lower()
    log(f"⏳ 开始实例化 SentenceTransformer...")
    model = SentenceTransformer(args.lm_model_or_path, trust_remote_code=trust_code, local_files_only=True)
    log(f"✅ SentenceTransformer 实例化完成")
    model.max_seq_length = args.max_seq_length
    log(f"⏳ 开始将模型加载到设备: {args.device} (首次加载 CUDA 会比较慢，请耐心等待...)")
    model.to(args.device)
    log(f"✅ 模型成功加载到设备: {args.device}")
    _wrap_data_parallel(model, enabled=args.multi_gpu)
    table_embeddings = [
        model.encode(sentences, show_progress_bar=True,
                     batch_size=args.batch_size, normalize_embeddings=True)
        for sentences in table_sentences]
    all_embeddings = list(chain(*table_embeddings))
    all_embeddings = np.array(all_embeddings)
    tm = timer.stop()
    log_time("encode all tables", tm)

    # 记录编码结果
    result_logger.log_encoding(
        embedding_shape=list(all_embeddings.shape),
        model_name=args.lm_model_or_path,
        time=tm,
        all_embeddings=all_embeddings,  # 传入嵌入向量
        table_sentences=table_sentences,  # 传入原始文本
        sample_size=5
    )

    # ========== 创建 tid 到实体名称的映射（用于输出评估详情）==========
    all_sentences_flat = list(chain(*table_sentences))
    tid_to_name = {tid: text for tid, text in enumerate(all_sentences_flat)}
    log(f"已创建 tid_to_name 映射，共 {len(tid_to_name)} 个实体")

    # 对比学习已移除，直接记录（未启用）
    result_logger.log_contrastive_learning(enabled=False)
    ground_truth = read_ground_truth(full_data_path)

    timer.start()

    # ========== 合并阶段：支持智能表配对 ==========
    if args.use_smart_pairing:
        # 智能表配对
        log(f"Using smart table pairing, strategy: {args.smart_pairing_strategy}")
        if args.run_in_parallel:
            table = merge_parallel_with_smart_pairing(tables, all_embeddings, args)
        else:
            table = merge_with_smart_pairing(tables, all_embeddings, args)
    else:
        # 原始随机配对
        log("Using random table pairing")
        if args.run_in_parallel:
            table = merge_parallel(tables, all_embeddings, args)
        else:
            table = merge(tables, all_embeddings, args)
    # ========== 合并阶段结束 ==========
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

    # ========== 输出合并后的表内容 ==========
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
    
    log(f"📝 合并后表内容已保存: {merge_table_file}")

    # 记录合并后的评估结果
    result_logger.log_evaluation(ground_truth, prediction, is_final=False)
    # 输出详细的实体组匹配信息（合并后）
    merge_output_file = f"logs/entity_groups_after_merge_{args.data_name}.txt"
    evaluate_log_with_output(ground_truth, prediction, tid_to_name, merge_output_file)

    # 记录最终评估结果
    result_logger.log_evaluation(ground_truth, prediction, is_final=True)
    # 输出详细的实体组匹配信息（最终）
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

    # ========== 保存所有中间结果 ==========
    json_file, txt_file = result_logger.save_results()
    log(f"中间结果已保存:")
    log(f"  JSON 文件: {json_file}")
    log(f"  文本文件: {txt_file}")

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
