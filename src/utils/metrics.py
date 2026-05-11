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
    在算 F1 的同时把预测分组按四类写出来，便于人工排查：
      1. 误匹配-其他（mis-match other）：包含错误成员或者完全错
      2. 误匹配-超集（mis-match superset）：覆盖了正确组但额外塞了别的
      3. 漏匹配（miss-match）：是正确组的真子集
      4. 完全正确（correct match）

    Args:
        ground_truth: 真实分组
        prediction: 预测分组
        tid_to_name: tid -> 文本
        output_file: 输出路径
    """
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)

    # 完全一致的
    correct_match = ground_truth_set.intersection(prediction_set)

    # 不一致的逐个分类
    incorrect_predictions = prediction_set - correct_match

    miss_match = set()           # 真子集
    mis_match_superset = set()   # 真超集
    mis_match_other = set()      # 其他

    # tid -> 它属于的 GT 组
    tid_to_gt_group = {}
    for gt_tuple in ground_truth:
        for tid in gt_tuple:
            tid_to_gt_group[tid] = gt_tuple

    for pred in incorrect_predictions:
        if not pred:
            continue

        pred_set = set(pred)

        # pred 里大多数实体来自哪个 GT 组
        gt_counts = {}
        for tid in pred:
            gt = tid_to_gt_group.get(tid)
            if gt:
                gt_id = id(gt)
                if gt_id not in gt_counts:
                    gt_counts[gt_id] = {'count': 0, 'gt': gt}
                gt_counts[gt_id]['count'] += 1

        if not gt_counts:
            # 全是噪声
            mis_match_other.add(pred)
            continue

        best_gt_id = max(gt_counts, key=lambda k: gt_counts[k]['count'])
        target_gt = gt_counts[best_gt_id]['gt']
        target_gt_set = set(target_gt)

        # 1) pred ⊂ gt：是真子集，漏匹配
        if pred_set.issubset(target_gt_set):
            # 已经排除了 == 的情况，所以这里一定是真子集
            miss_match.add(pred)

        # 2) gt ⊂ pred：是真超集，包对了但带了杂质
        elif target_gt_set.issubset(pred_set):
            mis_match_superset.add(pred)

        # 3) 其它
        else:
            mis_match_other.add(pred)

    with open(output_file, 'w', encoding='utf-8') as f:
        # 1) 误匹配 - 其他
        f.write("=== 误匹配 - 其他 (混合错误/噪声) ===\n")
        f.write(f"共 {len(mis_match_other)} 组\n\n")
        for tup in sorted(mis_match_other):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"预测分组: {names}\n")

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

        # 2) 误匹配 - 超集
        f.write("=== 误匹配 - 包含所有正确但有杂质 (Superset) ===\n")
        f.write(f"共 {len(mis_match_superset)} 组\n\n")
        for tup in sorted(mis_match_superset):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"预测分组 (超集): {names}\n")

            # 找到对应的完整 GT 组
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

                # 多出来的部分
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

        # 3) 漏匹配
        f.write("=== 漏匹配 (对但是缺/子集) ===\n")
        f.write(f"共 {len(miss_match)} 组\n\n")
        for tup in sorted(miss_match):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"预测分组 (子集): {names}\n")

            # 取第一个 tid 反查 GT
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

        # 4) 正确
        f.write("=== 正确匹配 ===\n")
        f.write(f"共 {len(correct_match)} 组\n\n")
        for tup in sorted(correct_match):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"{names}\n")

    log(f"实体组分类详细信息已保存到: {output_file}")

    # 算 P / R / F1
    truth = len(correct_match)
    P = truth / len(prediction_set) if len(prediction_set) > 0 else 0
    R = truth / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return Metric(p=P, r=R, f1=F1)


def evaluate_log_with_output(ground_truth: List[Tuple], prediction: List[Tuple],
                              tid_to_name: dict, output_file: str):
    """评估 + 把分组明细写到文件"""
    log(f"num of ground truth: {len(ground_truth)}")
    log(f"num of prediction: {len(prediction)}")

    f1_metric = evaluate_f1_with_output(ground_truth, prediction, tid_to_name, output_file)
    pair_f1_metric = evaluate_pair_f1(ground_truth, prediction)

    f1_metric.log(prefix="none")
    pair_f1_metric.log(prefix="pair")
