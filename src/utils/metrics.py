from dataclasses import dataclass
from itertools import combinations, chain
from typing import Optional, List, Tuple

from log import log


@dataclass()
class Metric:
    p: Optional[float] = None
    r: Optional[float] = None
    f1: Optional[float] = None

    def log(self, prefix=None):
        log(f"{f'[{prefix}] ' if prefix else ''}P={self.p:.4f}, R={self.r:.4f}, F1={self.f1:.4f}")


def evaluate_f1(ground_truth: List[Tuple], prediction: List[Tuple]) -> Metric:
    ground_truth = set(ground_truth)
    prediction = set(prediction)
    truth = len(ground_truth.intersection(prediction))
    P = truth / len(prediction) if len(prediction) > 0 else 0
    R = truth / len(ground_truth) if len(ground_truth) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return Metric(p=P, r=R, f1=F1)


def tuple_2_pairs(tuples: List[Tuple]) -> List[Tuple]:
    return list(chain(*[combinations(tup, 2) for tup in tuples]))


def evaluate_pair_f1(ground_truth: List[Tuple], prediction: List[Tuple]) -> Metric:
    ground_truth_pairs = tuple_2_pairs(ground_truth)
    prediction_pairs = tuple_2_pairs(prediction)
    return evaluate_f1(ground_truth_pairs, prediction_pairs)


def evaluate(ground_truth: List[Tuple], prediction: List[Tuple]) -> Tuple[Metric, Metric]:
    log(f"num of ground truth: {len(ground_truth)}")
    log(f"num of prediction: {len(prediction)}")
    f1_metric = evaluate_f1(ground_truth, prediction)
    pair_f1_metric = evaluate_pair_f1(ground_truth, prediction)
    return f1_metric, pair_f1_metric


def evaluate_log(ground_truth: List[Tuple], prediction: List[Tuple]):
    metric, pair_metric = evaluate(ground_truth, prediction)
    metric.log(prefix="none")
    pair_metric.log(prefix="pair")


def evaluate_f1_with_output(ground_truth: List[Tuple], prediction: List[Tuple],
                             tid_to_name: dict, output_file: str) -> Metric:
    """
    评估F1分数并输出实体组分类信息到文件
    分类：
    1. 误匹配-其他 (Mis-match Other): 混合了错误答案或完全错误
    2. 误匹配-包含正确但有杂质 (Mis-match Superset): 包含了所有正确答案但多出了错误答案 (Impure Superset)
    3. 漏匹配 (Miss-match): 属于正确答案的真子集 (Pure Subset)
    4. 正确匹配 (Correct match): 完全一致

    Args:
        ground_truth: 标准答案实体组列表
        prediction: 预测的实体组列表
        tid_to_name: tid到实体名称（文本）的映射字典
        output_file: 输出文件路径

    Returns:
        Metric: 包含P, R, F1的评估指标
    """
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)

    # 1. 正确匹配 (Correct match): P == GT
    correct_match = ground_truth_set.intersection(prediction_set)
    
    # 找出所有预测中不完全匹配的项
    incorrect_predictions = prediction_set - correct_match
    
    miss_match = set()           # P < GT (Pure Subset)
    mis_match_superset = set()   # P > GT (Impure Superset)
    mis_match_other = set()      # Other
    
    # 建立 tid -> ground_truth 分组的映射
    tid_to_gt_group = {}
    for gt_tuple in ground_truth:
        for tid in gt_tuple:
            tid_to_gt_group[tid] = gt_tuple

    for pred in incorrect_predictions:
        if not pred:
            continue
            
        pred_set = set(pred)
        
        # 找到 pred 可能对应的 GT 组
        # 策略：找到 pred 中包含最多的那个 GT 组
        gt_counts = {}
        for tid in pred:
            gt = tid_to_gt_group.get(tid)
            if gt:
                gt_id = id(gt) # Use ID as dictionary key
                if gt_id not in gt_counts:
                    gt_counts[gt_id] = {'count': 0, 'gt': gt}
                gt_counts[gt_id]['count'] += 1
        
        if not gt_counts:
            # 没有任何实体在 GT 中 -> 完全噪声 -> Other
            mis_match_other.add(pred)
            continue
            
        # 找到 pred 中占比最大的 GT
        best_gt_id = max(gt_counts, key=lambda k: gt_counts[k]['count'])
        target_gt = gt_counts[best_gt_id]['gt']
        target_gt_set = set(target_gt)
        
        # 逻辑判断
        # 1. 检查是否是 Pure Subset (Miss-match)
        # 条件：pred 是 target_gt 的子集，且 pred 中没有不属于 target_gt 的实体
        if pred_set.issubset(target_gt_set):
            # 已经是 incorrect_predictions，所以肯定不是 equal，那就是 proper subset
            miss_match.add(pred)
            
        # 2. 检查是否是 Impure Superset (Mis-match Superset)
        # 条件：target_gt 是 pred 的子集 (即 pred 包含了 target_gt 的所有实体)
        elif target_gt_set.issubset(pred_set):
            mis_match_superset.add(pred)
            
        # 3. 其他情况 (Other)
        else:
            mis_match_other.add(pred)

    # 输出结果
    with open(output_file, 'w', encoding='utf-8') as f:
        # === 1. 误匹配 - 其他 (包含了错误答案/杂质/完全错误) ===
        f.write("=== 误匹配 - 其他 (混合错误/噪声) ===\n")
        f.write(f"共 {len(mis_match_other)} 组\n\n")
        for tup in sorted(mis_match_other):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"预测分组: {names}\n")
            
            # 详细分析
            f.write("详细分析:\n")
            seen_gts = {}
            for tid in tup:
                gt_group = tid_to_gt_group.get(tid)
                entity_name = tid_to_name.get(tid, f"tid_{tid}")
                if gt_group:
                    gt_id = id(gt_group)
                    if gt_id not in seen_gts:
                        gt_names = [tid_to_name.get(t, f"tid_{t}") for t in gt_group]
                        seen_gts[gt_id] = gt_names
                    f.write(f"  - {entity_name} -> 属于GT组: {seen_gts[gt_id]}\n")
                else:
                    f.write(f"  - {entity_name} -> 不在标准答案中 (噪声)\n")
            f.write("\n")
        f.write("="*70 + "\n\n")
        
        # === 2. 误匹配 - 包含正确但有杂质 (Superset) ===
        f.write("=== 误匹配 - 包含所有正确但有杂质 (Superset) ===\n")
        f.write(f"共 {len(mis_match_superset)} 组\n\n")
        for tup in sorted(mis_match_superset):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"预测分组 (超集): {names}\n")
            
            # 找到被包含的完整 GT
            # 同样逻辑找到 target_gt
            gt_counts = {}
            for tid in tup:
                gt = tid_to_gt_group.get(tid)
                if gt:
                    gt_id = id(gt)
                    if gt_id not in gt_counts:
                        gt_counts[gt_id] = {'count': 0, 'gt': gt}
                    gt_counts[gt_id]['count'] += 1
            if gt_counts:
                best_gt_id = max(gt_counts, key=lambda k: gt_counts[k]['count'])
                target_gt = gt_counts[best_gt_id]['gt']
                gt_names = [tid_to_name.get(t, f"tid_{t}") for t in target_gt]
                f.write(f"包含了完整GT组: {gt_names}\n")
                
                # 找出多余的实体
                extra_tids = set(tup) - set(target_gt)
                for tid in extra_tids:
                    entity_name = tid_to_name.get(tid, f"tid_{tid}")
                    gt_group = tid_to_gt_group.get(tid)
                    if gt_group:
                        gt_names = [tid_to_name.get(t, f"tid_{t}") for t in gt_group]
                        f.write(f"  - 多余实体: {entity_name} (属于其他GT: {gt_names})\n")
                    else:
                        f.write(f"  - 多余实体: {entity_name} (噪声)\n")
            f.write("\n")
        f.write("="*70 + "\n\n")

        # === 3. 漏匹配 (对但是缺/子集) ===
        f.write("=== 漏匹配 (对但是缺/子集) ===\n")
        f.write(f"共 {len(miss_match)} 组\n\n")
        for tup in sorted(miss_match):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"预测分组 (子集): {names}\n")
            
            # 找到对应的完整GT
            first_tid = tup[0]
            gt_group = tid_to_gt_group.get(first_tid)
            if gt_group:
                gt_names = [tid_to_name.get(t, f"tid_{t}") for t in gt_group]
                missing_tids = set(gt_group) - set(tup)
                missing_names = [tid_to_name.get(t, f"tid_{t}") for t in missing_tids]
                f.write(f"完整GT组: {gt_names}\n")
                f.write(f"缺失实体: {missing_names}\n")
            f.write("\n")
        f.write("="*70 + "\n\n")

        # === 4. 正确匹配 ===
        f.write("=== 正确匹配 ===\n")
        f.write(f"共 {len(correct_match)} 组\n\n")
        for tup in sorted(correct_match):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"{names}\n")

    log(f"实体组分类详细信息已保存到: {output_file}")

    # 计算指标
    truth = len(correct_match)
    P = truth / len(prediction_set) if len(prediction_set) > 0 else 0
    R = truth / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return Metric(p=P, r=R, f1=F1)


def evaluate_log_with_output(ground_truth: List[Tuple], prediction: List[Tuple],
                              tid_to_name: dict, output_file: str):
    """
    评估并记录日志，同时输出详细的实体组信息到文件

    Args:
        ground_truth: 标准答案实体组列表
        prediction: 预测的实体组列表
        tid_to_name: tid到实体名称（文本）的映射字典
        output_file: 输出文件路径
    """
    log(f"num of ground truth: {len(ground_truth)}")
    log(f"num of prediction: {len(prediction)}")

    # 使用带输出功能的评估函数
    f1_metric = evaluate_f1_with_output(ground_truth, prediction, tid_to_name, output_file)
    pair_f1_metric = evaluate_pair_f1(ground_truth, prediction)

    f1_metric.log(prefix="none")
    pair_f1_metric.log(prefix="pair")
