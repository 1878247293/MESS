"""
Evaluation metrics.

- `evaluate_f1` -- P/R/F1 at the tuple-set level.
- `evaluate_pair_f1` -- break tuples into pairwise combinations and compute F1, measuring whether the right items got matched.
- `evaluate_f1_with_output` -- while computing F1, also write the predicted groups in four categories
  (exact correct / missed match / mis-match-superset / mis-match-other) to a txt for manual inspection.
"""

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
    While computing F1, write out the predicted groups in four categories for manual inspection:
      1. mis-match other: contains wrong members or is entirely wrong
      2. mis-match superset: covers the correct group but includes extra items
      3. miss-match: a proper subset of the correct group
      4. correct match

    Args:
        ground_truth: the true groups
        prediction: the predicted groups
        tid_to_name: tid -> text
        output_file: output path
    """
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)

    # exact matches
    correct_match = ground_truth_set.intersection(prediction_set)

    # classify the mismatches one by one
    incorrect_predictions = prediction_set - correct_match

    miss_match = set()           # proper subset
    mis_match_superset = set()   # proper superset
    mis_match_other = set()      # other

    # tid -> the GT group it belongs to
    tid_to_gt_group = {}
    for gt_tuple in ground_truth:
        for tid in gt_tuple:
            tid_to_gt_group[tid] = gt_tuple

    for pred in incorrect_predictions:
        if not pred:
            continue

        pred_set = set(pred)

        # which GT group most entities in pred come from
        gt_counts = {}
        for tid in pred:
            gt = tid_to_gt_group.get(tid)
            if gt:
                gt_id = id(gt)
                if gt_id not in gt_counts:
                    gt_counts[gt_id] = {'count': 0, 'gt': gt}
                gt_counts[gt_id]['count'] += 1

        if not gt_counts:
            # all noise
            mis_match_other.add(pred)
            continue

        best_gt_id = max(gt_counts, key=lambda k: gt_counts[k]['count'])
        target_gt = gt_counts[best_gt_id]['gt']
        target_gt_set = set(target_gt)

        # 1) pred is a subset of gt: a proper subset, a missed match
        if pred_set.issubset(target_gt_set):
            # the == case is already excluded, so this is definitely a proper subset
            miss_match.add(pred)

        # 2) gt is a subset of pred: a proper superset, correct but with impurities
        elif target_gt_set.issubset(pred_set):
            mis_match_superset.add(pred)

        # 3) other
        else:
            mis_match_other.add(pred)

    with open(output_file, 'w', encoding='utf-8') as f:
        # 1) mis-match - other
        f.write("=== Mis-match - other (mixed errors/noise) ===\n")
        f.write(f"{len(mis_match_other)} groups in total\n\n")
        for tup in sorted(mis_match_other):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"Predicted group: {names}\n")

            f.write("Detailed analysis:\n")
            seen_gts = {}
            for tid in tup:
                gt_group = tid_to_gt_group.get(tid)
                entity_name = tid_to_name.get(tid, f"tid_{tid}")
                if gt_group:
                    gt_id = id(gt_group)
                    if gt_id not in seen_gts:
                        gt_names = [tid_to_name.get(t, f"tid_{t}") for t in gt_group]
                        seen_gts[gt_id] = gt_names
                    f.write(f"  - {entity_name} -> belongs to GT group: {seen_gts[gt_id]}\n")
                else:
                    f.write(f"  - {entity_name} -> not in the ground truth (noise)\n")
            f.write("\n")
        f.write("="*70 + "\n\n")

        # 2) mis-match - superset
        f.write("=== Mis-match - contains all correct items but has impurities (Superset) ===\n")
        f.write(f"{len(mis_match_superset)} groups in total\n\n")
        for tup in sorted(mis_match_superset):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"Predicted group (superset): {names}\n")

            # find the corresponding complete GT group
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
                f.write(f"Contains the complete GT group: {gt_names}\n")

                # the extra part
                extra_tids = set(tup) - set(target_gt)
                for tid in extra_tids:
                    entity_name = tid_to_name.get(tid, f"tid_{tid}")
                    gt_group = tid_to_gt_group.get(tid)
                    if gt_group:
                        gt_names = [tid_to_name.get(t, f"tid_{t}") for t in gt_group]
                        f.write(f"  - extra entity: {entity_name} (belongs to another GT: {gt_names})\n")
                    else:
                        f.write(f"  - extra entity: {entity_name} (noise)\n")
            f.write("\n")
        f.write("="*70 + "\n\n")

        # 3) missed match
        f.write("=== Missed match (correct but incomplete/subset) ===\n")
        f.write(f"{len(miss_match)} groups in total\n\n")
        for tup in sorted(miss_match):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"Predicted group (subset): {names}\n")

            # use the first tid to look up the GT
            first_tid = tup[0]
            gt_group = tid_to_gt_group.get(first_tid)
            if gt_group:
                gt_names = [tid_to_name.get(t, f"tid_{t}") for t in gt_group]
                missing_tids = set(gt_group) - set(tup)
                missing_names = [tid_to_name.get(t, f"tid_{t}") for t in missing_tids]
                f.write(f"Complete GT group: {gt_names}\n")
                f.write(f"Missing entities: {missing_names}\n")
            f.write("\n")
        f.write("="*70 + "\n\n")

        # 4) correct
        f.write("=== Correct matches ===\n")
        f.write(f"{len(correct_match)} groups in total\n\n")
        for tup in sorted(correct_match):
            names = [tid_to_name.get(tid, f"tid_{tid}") for tid in tup]
            f.write(f"{names}\n")

    log(f"Detailed entity-group classification saved to: {output_file}")

    # compute P / R / F1
    truth = len(correct_match)
    P = truth / len(prediction_set) if len(prediction_set) > 0 else 0
    R = truth / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    return Metric(p=P, r=R, f1=F1)


def evaluate_log_with_output(ground_truth: List[Tuple], prediction: List[Tuple],
                              tid_to_name: dict, output_file: str):
    """Evaluate + write the group details to a file"""
    log(f"num of ground truth: {len(ground_truth)}")
    log(f"num of prediction: {len(prediction)}")

    f1_metric = evaluate_f1_with_output(ground_truth, prediction, tid_to_name, output_file)
    pair_f1_metric = evaluate_pair_f1(ground_truth, prediction)

    f1_metric.log(prefix="none")
    pair_f1_metric.log(prefix="pair")
