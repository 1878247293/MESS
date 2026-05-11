"""
中间结果落盘。
跑一次主流程会把每个阶段的输入输出 + 指标都收集下来，最后写一份 JSON 和一份易读的 TXT。
"""
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class RunInfo:
    timestamp: str
    dataset: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributeSelectionResult:
    selected_attrs: List[str] = field(default_factory=list)
    attribute_scores: Dict[str, float] = field(default_factory=dict)
    time: float = 0.0
    from_cache: bool = False


@dataclass
class DataLoadingResult:
    num_tables: int = 0
    table_sizes: List[int] = field(default_factory=list)
    total_entities: int = 0
    time: float = 0.0
    table_samples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EncodingResult:
    embedding_shape: List[int] = field(default_factory=list)
    model_name: str = ""
    time: float = 0.0
    embedding_samples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ContrastiveLearningResult:
    enabled: bool = False
    epochs: int = 0
    time: float = 0.0


@dataclass
class MergeInfo:
    tables: List[str] = field(default_factory=list)
    pairs: int = 0
    time: float = 0.0
    merge_examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HierarchyLevel:
    level: int = 0
    num_tables: int = 0
    num_tuples_before: int = 0
    num_tuples_after: int = 0
    merges: List[MergeInfo] = field(default_factory=list)


@dataclass
class MergingResult:
    hierarchy_levels: List[HierarchyLevel] = field(default_factory=list)
    total_time: float = 0.0
    final_tuples: int = 0


@dataclass
class EvaluationResult:
    num_ground_truth: int = 0
    num_prediction: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    pair_precision: float = 0.0
    pair_recall: float = 0.0
    pair_f1: float = 0.0


@dataclass
class PruningResult:
    tuples_before: int = 0
    tuples_after: int = 0
    tuples_removed: int = 0
    entities_before: int = 0
    entities_after: int = 0
    entities_removed: int = 0
    removal_rate: float = 0.0
    time: float = 0.0
    tuple_size_dist_before: Dict[int, int] = field(default_factory=dict)
    tuple_size_dist_after: Dict[int, int] = field(default_factory=dict)
    largest_tuple_before: int = 0
    largest_tuple_after: int = 0
    avg_tuple_size_before: float = 0.0
    avg_tuple_size_after: float = 0.0
    pruning_examples: List[Dict[str, Any]] = field(default_factory=list)


class ResultLogger:
    def __init__(self, dataset_name: str, output_dir: str = "results"):
        self.dataset_name = dataset_name
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.run_info = RunInfo(timestamp=self.timestamp, dataset=dataset_name)
        self.attribute_selection = AttributeSelectionResult()
        self.data_loading = DataLoadingResult()
        self.encoding = EncodingResult()
        self.contrastive_learning = ContrastiveLearningResult()
        self.merging = MergingResult()
        self.evaluation_after_merge = EvaluationResult()
        self.pruning = PruningResult()
        self.evaluation_final = EvaluationResult()

        # 当前正在记录的层（合并阶段会逐层切换）
        self.current_hierarchy_level: Optional[HierarchyLevel] = None

        # 杂项：LLM Selecting 之类的额外数据
        self.data: Dict[str, Any] = {}

    def log_resource_usage(self, stage_name: str, stage_info: Dict[str, Any]):
        resource_usage = self.data.setdefault("resource_usage", {})
        resource_usage[stage_name] = stage_info

    def log_run_summary(self, summary_info: Dict[str, Any]):
        self.data["run_summary"] = summary_info

    def set_parameters(self, args):
        self.run_info.parameters = {
            k: v for k, v in args.__dict__.items()
            if not str(k).startswith("__")
        }

    def log_attribute_selection(self, selected_attrs: List[str],
                                attribute_scores: Dict[str, float],
                                time: float, from_cache: bool = False):
        self.attribute_selection.selected_attrs = selected_attrs
        self.attribute_selection.attribute_scores = attribute_scores
        self.attribute_selection.time = time
        self.attribute_selection.from_cache = from_cache

    def log_data_loading(self, num_tables: int, table_sizes: List[int], time: float,
                        tables_df: List = None, sample_size: int = 5):
        self.data_loading.num_tables = num_tables
        self.data_loading.table_sizes = table_sizes
        self.data_loading.total_entities = sum(table_sizes)
        self.data_loading.time = time

        # 每张表抽前 N 行存下来，落盘后能直接看
        if tables_df is not None:
            for idx, table_df in enumerate(tables_df):
                sample_records = []
                for i in range(min(sample_size, len(table_df))):
                    record = table_df.iloc[i].to_dict()
                    # numpy 类型 JSON 不能直接序列化
                    record = {k: self._convert_to_python_type(v) for k, v in record.items()}
                    sample_records.append(record)

                self.data_loading.table_samples.append({
                    'table_id': idx,
                    'columns': list(table_df.columns),
                    'sample_records': sample_records
                })

    @staticmethod
    def _convert_to_python_type(val):
        """把 numpy 标量降成 python 原生类型，方便 json.dump"""
        if isinstance(val, (np.integer, np.int64, np.int32, np.uint64, np.uint32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (np.bool_, bool)):
            return bool(val)
        else:
            return val

    def log_encoding(self, embedding_shape: List[int], model_name: str, time: float,
                    all_embeddings=None, table_sentences=None, sample_size: int = 5):
        self.encoding.embedding_shape = embedding_shape
        self.encoding.model_name = model_name
        self.encoding.time = time

        if all_embeddings is not None and table_sentences is not None:
            from itertools import chain
            all_sentences_flat = list(chain(*table_sentences)) if isinstance(table_sentences[0], list) else table_sentences

            for i in range(min(sample_size, len(all_embeddings))):
                self.encoding.embedding_samples.append({
                    'entity_id': i,
                    'text': all_sentences_flat[i] if i < len(all_sentences_flat) else 'N/A',
                    'embedding_vector': all_embeddings[i].tolist()[:10],  # 只留前 10 维
                    'embedding_norm': float(np.linalg.norm(all_embeddings[i]))
                })

    def log_contrastive_learning(self, enabled: bool, epochs: int = 0, time: float = 0.0):
        self.contrastive_learning.enabled = enabled
        self.contrastive_learning.epochs = epochs
        self.contrastive_learning.time = time

    def start_hierarchy_level(self, level: int, num_tables: int, num_tuples: int = 0):
        self.current_hierarchy_level = HierarchyLevel(
            level=level,
            num_tables=num_tables,
            num_tuples_before=num_tuples
        )

    def log_merge(self, table_i_idx: str, table_j_idx: str, pairs: int, merge_time: float,
                 merge_examples: List[Tuple] = None):
        if self.current_hierarchy_level is not None:
            examples = []
            if merge_examples is not None:
                # 最多留 5 条
                for example in merge_examples[:min(5, len(merge_examples))]:
                    examples.append({
                        'entity_i': example[0],
                        'entity_j': example[1],
                        'distance': float(example[2]) if len(example) > 2 and example[2] is not None else None
                    })

            merge_info = MergeInfo(
                tables=[table_i_idx, table_j_idx],
                pairs=pairs,
                time=merge_time,
                merge_examples=examples
            )
            self.current_hierarchy_level.merges.append(merge_info)

    def finish_hierarchy_level(self, num_tuples_after: int = 0):
        if self.current_hierarchy_level is not None:
            self.current_hierarchy_level.num_tuples_after = num_tuples_after
            self.merging.hierarchy_levels.append(self.current_hierarchy_level)
            self.current_hierarchy_level = None

    def log_merging_summary(self, total_time: float, final_tuples: int):
        self.merging.total_time = total_time
        self.merging.final_tuples = final_tuples

    def log_evaluation(self, ground_truth: List[Tuple], prediction: List[Tuple],
                      is_final: bool = False):
        from metrics import evaluate

        metric, pair_metric = evaluate(ground_truth, prediction)

        result = EvaluationResult(
            num_ground_truth=len(ground_truth),
            num_prediction=len(prediction),
            precision=metric.p,
            recall=metric.r,
            f1=metric.f1,
            pair_precision=pair_metric.p,
            pair_recall=pair_metric.r,
            pair_f1=pair_metric.f1
        )

        if is_final:
            self.evaluation_final = result
        else:
            self.evaluation_after_merge = result

    @staticmethod
    def _compute_tuple_stats(tuples: List[Tuple]):
        """size 分布、最大组、平均组大小"""
        if not tuples:
            return {}, 0, 0.0

        size_dist = {}
        for t in tuples:
            size = len(t)
            size_dist[size] = size_dist.get(size, 0) + 1

        largest = max(len(t) for t in tuples)

        avg_size = sum(len(t) for t in tuples) / len(tuples)

        return size_dist, largest, avg_size

    def log_pruning(self, prediction_before: List[Tuple], prediction_after: List[Tuple], time: float):
        """传剪枝前后的完整 tuple 列表，里面会顺手算分布和差异样本"""
        tuples_before = len(prediction_before)
        tuples_after = len(prediction_after)
        entities_before = sum(len(item) for item in prediction_before)
        entities_after = sum(len(item) for item in prediction_after)

        size_dist_before, largest_before, avg_size_before = self._compute_tuple_stats(prediction_before)
        size_dist_after, largest_after, avg_size_after = self._compute_tuple_stats(prediction_after)

        # 用 sorted 元组作 key 比较
        before_set = {tuple(sorted(t)) for t in prediction_before}
        after_set = {tuple(sorted(t)) for t in prediction_after}

        removed_tuples = before_set - after_set
        modified_tuples = []

        before_map = {tuple(sorted(t)): t for t in prediction_before}
        after_map = {tuple(sorted(t)): t for t in prediction_after}

        for t_before in prediction_before:
            t_sorted = tuple(sorted(t_before))
            if t_sorted in after_map:
                t_after = after_map[t_sorted]
                if len(t_before) != len(t_after):
                    modified_tuples.append({
                        'before': list(t_before),
                        'after': list(t_after),
                        'entities_removed': len(t_before) - len(t_after)
                    })

        # 各类样本最多记 5 条
        for removed_tuple in list(removed_tuples)[:5]:
            self.pruning.pruning_examples.append({
                'type': 'removed',
                'tuple': list(removed_tuple),
                'size': len(removed_tuple)
            })

        for modified in modified_tuples[:5]:
            self.pruning.pruning_examples.append({
                'type': 'modified',
                'before': modified['before'],
                'after': modified['after'],
                'entities_removed': modified['entities_removed']
            })

        self.pruning.tuples_before = tuples_before
        self.pruning.tuples_after = tuples_after
        self.pruning.tuples_removed = tuples_before - tuples_after
        self.pruning.entities_before = entities_before
        self.pruning.entities_after = entities_after
        self.pruning.entities_removed = entities_before - entities_after
        self.pruning.removal_rate = (
            self.pruning.entities_removed / entities_before * 100
            if entities_before > 0 else 0.0
        )
        self.pruning.time = time
        self.pruning.tuple_size_dist_before = size_dist_before
        self.pruning.tuple_size_dist_after = size_dist_after
        self.pruning.largest_tuple_before = largest_before
        self.pruning.largest_tuple_after = largest_after
        self.pruning.avg_tuple_size_before = avg_size_before
        self.pruning.avg_tuple_size_after = avg_size_after

    def save_results(self):
        """JSON + TXT 各写一份"""
        base_filename = f"{self.timestamp}_{self.dataset_name}"

        json_file = self.output_dir / f"{base_filename}_results.json"
        self._save_json(json_file)

        txt_file = self.output_dir / f"{base_filename}_intermediate.txt"
        self._save_txt(txt_file)

        return str(json_file), str(txt_file)

    def _save_json(self, filepath: Path):
        results = {
            "run_info": asdict(self.run_info),
            "attribute_selection": asdict(self.attribute_selection),
            "data_loading": asdict(self.data_loading),
            "encoding": asdict(self.encoding),
            "contrastive_learning": asdict(self.contrastive_learning),
            "merging": asdict(self.merging),
            "evaluation_after_merge": asdict(self.evaluation_after_merge),
            "pruning": asdict(self.pruning),
            "evaluation_final": asdict(self.evaluation_final),
            "extra_data": self.data,
        }

        # 递归把 numpy 类型替换成 python 原生，否则 json.dump 会爆
        results = self._convert_numpy_types(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {self._convert_to_python_type(k): self._convert_numpy_types(v)
                    for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return self._convert_to_python_type(obj)

    def _save_txt(self, filepath: Path):
        lines = []

        lines.append("=" * 80)
        lines.append(f"运行中间结果")
        lines.append(f"数据集: {self.dataset_name}")
        lines.append(f"时间戳: {self.timestamp}")
        lines.append("=" * 80)
        lines.append("")

        # 参数
        lines.append("【运行参数】")
        lines.append("-" * 80)
        for k, v in self.run_info.parameters.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

        # 属性选择
        lines.append("【属性选择阶段】")
        lines.append("-" * 80)
        if self.attribute_selection.from_cache:
            lines.append("  状态: 命中缓存")
        else:
            lines.append("  状态: 实跑")
        lines.append(f"  耗时: {self.attribute_selection.time:.4f} 秒")
        lines.append(f"  selected_attrs: {self.attribute_selection.selected_attrs}")
        lines.append("  各列 sim:")
        for attr, score in self.attribute_selection.attribute_scores.items():
            status = "[keep]" if attr in self.attribute_selection.selected_attrs else "[drop]"
            lines.append(f"    {attr}: {score:.4f} {status}")
        lines.append("")

        resource_usage = self.data.get("resource_usage", {})
        if resource_usage:
            lines.append("【阶段资源统计】")
            lines.append("-" * 80)
            for stage_name, info in resource_usage.items():
                lines.append(f"  {stage_name}:")
                lines.append(f"    总耗时: {info.get('total_time', 0.0):.4f} 秒")
                lines.append(f"    峰值内存: {info.get('peak_memory_mb', 0.0):.2f} MB")
                lines.append(f"    峰值显存: {info.get('peak_gpu_memory_mb', 0.0):.2f} MB")
                lines.append(f"    峰值字节数: {info.get('peak_memory_bytes', 0)}")
                lines.append(f"    峰值显存字节数: {info.get('peak_gpu_memory_bytes', 0)}")
                lines.append(f"    采样次数: {info.get('invocations', 0)}")
                lines.append(f"    内存采样可用: {info.get('available', False)}")
                lines.append(f"    显存采样可用: {info.get('gpu_available', False)}")
            lines.append("")

        run_summary = self.data.get("run_summary")
        if run_summary:
            lines.append("【本次运行总结】")
            lines.append("-" * 80)
            for phase_name, phase_time in run_summary.get("phase_times", {}).items():
                lines.append(f"  {phase_name}: {phase_time:.4f} 秒")
            lines.append(f"  总时间: {run_summary.get('total_time', 0.0):.4f} 秒")
            lines.append(f"  最大内存: {run_summary.get('peak_memory_human', 'N/A')}")
            lines.append(f"  最大显存: {run_summary.get('peak_gpu_memory_human', 'N/A')}")
            lines.append("")

        # 数据加载
        lines.append("【数据加载阶段】")
        lines.append("-" * 80)
        lines.append(f"  耗时: {self.data_loading.time:.4f} 秒")
        lines.append(f"  表数量: {self.data_loading.num_tables}")
        lines.append(f"  总实体数: {self.data_loading.total_entities}")
        lines.append(f"  各表大小: {self.data_loading.table_sizes}")

        if self.data_loading.table_samples:
            lines.append("")
            lines.append("  【表数据样本】")
            for sample in self.data_loading.table_samples:
                lines.append(f"    表 {sample['table_id']}:")
                lines.append(f"      列名: {sample['columns']}")
                lines.append(f"      样本 (前5条):")
                for i, record in enumerate(sample['sample_records']):
                    lines.append(f"        记录 {i+1}: {record}")
        lines.append("")

        # 编码
        lines.append("【编码阶段】")
        lines.append("-" * 80)
        lines.append(f"  耗时: {self.encoding.time:.4f} 秒")
        lines.append(f"  模型: {self.encoding.model_name}")
        lines.append(f"  嵌入维度: {self.encoding.embedding_shape}")

        if self.encoding.embedding_samples:
            lines.append("")
            lines.append("  【编码样本】(前5条)")
            for sample in self.encoding.embedding_samples:
                lines.append(f"    实体 {sample['entity_id']}:")
                lines.append(f"      文本: {sample['text']}")
                lines.append(f"      嵌入向量(前10维): {[f'{v:.4f}' for v in sample['embedding_vector']]}")
                lines.append(f"      向量模长: {sample['embedding_norm']:.4f}")
        lines.append("")

        # 对比学习
        if self.contrastive_learning.enabled:
            lines.append("【对比学习阶段】")
            lines.append("-" * 80)
            lines.append(f"  耗时: {self.contrastive_learning.time:.4f} 秒")
            lines.append(f"  训练轮数: {self.contrastive_learning.epochs}")
            lines.append("")

        # 合并
        lines.append("【合并阶段】")
        lines.append("-" * 80)
        lines.append(f"  总耗时: {self.merging.total_time:.4f} 秒")
        lines.append(f"  最终元组数: {self.merging.final_tuples}")
        lines.append(f"  各层详情:")
        for level_info in self.merging.hierarchy_levels:
            lines.append(f"    Level {level_info.level}: {level_info.num_tables} 个表")
            if level_info.num_tuples_before > 0:
                lines.append(f"      合并前元组数: {level_info.num_tuples_before}")
            if level_info.num_tuples_after > 0:
                lines.append(f"      合并后元组数: {level_info.num_tuples_after}")
                if level_info.num_tuples_before > 0:
                    reduction = level_info.num_tuples_before - level_info.num_tuples_after
                    reduction_rate = (reduction / level_info.num_tuples_before) * 100
                    lines.append(f"      元组减少: {reduction} ({reduction_rate:.2f}%)")
            for merge in level_info.merges:
                lines.append(f"      - 合并 {merge.tables[0]} & {merge.tables[1]}: "
                           f"{merge.pairs} 对, {merge.time:.4f}s")
                if merge.merge_examples:
                    lines.append(f"        示例:")
                    for ex in merge.merge_examples:
                        dist_str = f", dist={ex['distance']:.4f}" if ex['distance'] is not None else ""
                        lines.append(f"          实体 {ex['entity_i']} <-> 实体 {ex['entity_j']}{dist_str}")
        lines.append("")

        # 合并后评估
        lines.append("【合并后评估】")
        lines.append("-" * 80)
        self._format_evaluation(lines, self.evaluation_after_merge)
        lines.append("")

        # 剪枝
        lines.append("【剪枝阶段】")
        lines.append("-" * 80)
        lines.append(f"  耗时: {self.pruning.time:.4f} 秒")
        lines.append(f"  剪枝前元组数: {self.pruning.tuples_before}")
        lines.append(f"  剪枝后元组数: {self.pruning.tuples_after}")
        lines.append(f"  移除元组数: {self.pruning.tuples_removed}")
        lines.append(f"  剪枝前实体数: {self.pruning.entities_before}")
        lines.append(f"  剪枝后实体数: {self.pruning.entities_after}")
        lines.append(f"  移除实体数: {self.pruning.entities_removed}")
        lines.append(f"  移除率: {self.pruning.removal_rate:.2f}%")
        lines.append("")

        if self.pruning.tuple_size_dist_before:
            lines.append("  元组大小分布:")
            lines.append(f"    剪枝前:")
            lines.append(f"      平均: {self.pruning.avg_tuple_size_before:.2f}")
            lines.append(f"      最大: {self.pruning.largest_tuple_before}")
            lines.append(f"      分布:")
            for size in sorted(self.pruning.tuple_size_dist_before.keys()):
                count = self.pruning.tuple_size_dist_before[size]
                lines.append(f"        size {size}: {count}")

            lines.append(f"    剪枝后:")
            lines.append(f"      平均: {self.pruning.avg_tuple_size_after:.2f}")
            lines.append(f"      最大: {self.pruning.largest_tuple_after}")
            lines.append(f"      分布:")
            for size in sorted(self.pruning.tuple_size_dist_after.keys()):
                count = self.pruning.tuple_size_dist_after[size]
                count_before = self.pruning.tuple_size_dist_before.get(size, 0)
                change = count - count_before
                change_str = f" ({change:+d})" if change != 0 else ""
                lines.append(f"        size {size}: {count}{change_str}")

        if self.pruning.pruning_examples:
            lines.append("")
            lines.append("  【剪枝示例】")
            for example in self.pruning.pruning_examples:
                if example['type'] == 'removed':
                    lines.append(f"    [removed] (size={example['size']}): {example['tuple']}")
                elif example['type'] == 'modified':
                    lines.append(f"    [modified]:")
                    lines.append(f"      before: {example['before']}")
                    lines.append(f"      after : {example['after']}")
                    lines.append(f"      entities_removed: {example['entities_removed']}")
        lines.append("")

        # 终评估
        lines.append("【最终评估】")
        lines.append("-" * 80)
        self._format_evaluation(lines, self.evaluation_final)
        lines.append("")

        # 提升幅度
        lines.append("【性能提升】")
        lines.append("-" * 80)
        f1_improvement = (self.evaluation_final.f1 - self.evaluation_after_merge.f1) * 100
        pair_f1_improvement = (self.evaluation_final.pair_f1 - self.evaluation_after_merge.pair_f1) * 100
        lines.append(f"  F1 提升: {f1_improvement:+.2f}%")
        lines.append(f"  Pair-F1 提升: {pair_f1_improvement:+.2f}%")
        lines.append("")

        lines.append("=" * 80)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _format_evaluation(self, lines: List[str], eval_result: EvaluationResult):
        lines.append(f"  真实元组数: {eval_result.num_ground_truth}")
        lines.append(f"  预测元组数: {eval_result.num_prediction}")
        lines.append(f"  Tuple-level:")
        lines.append(f"    Precision: {eval_result.precision:.4f}")
        lines.append(f"    Recall:    {eval_result.recall:.4f}")
        lines.append(f"    F1:        {eval_result.f1:.4f}")
        lines.append(f"  Pair-level:")
        lines.append(f"    Precision: {eval_result.pair_precision:.4f}")
        lines.append(f"    Recall:    {eval_result.pair_recall:.4f}")
        lines.append(f"    F1:        {eval_result.pair_f1:.4f}")
