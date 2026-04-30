"""前端辅助:数据集统计、日志解析、结果读取、参数预设。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# tqdm 进度条行:"Loading:  10%|█  | 20/199 [..."
_PROGRESS_RE = re.compile(r"^(.*?)\s*\d+%\|")
# 百分比:"...  35%"
_PCT_RE = re.compile(r"(\d{1,3})%")
# [phase X/Y] 阶段标记(若后端打印)
_PHASE_RE = re.compile(r"\[phase\s+(\d+)/(\d+)\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 日志整理
# ---------------------------------------------------------------------------
def collapse_progress_lines(log_text: str) -> str:
    """合并同一 tqdm 进度条的连续输出,只保留最后一行。"""
    lines = log_text.split("\n")
    kept: list[str] = []
    for line in lines:
        m = _PROGRESS_RE.match(line)
        if m and kept:
            prev = _PROGRESS_RE.match(kept[-1])
            if prev and prev.group(1).strip() == m.group(1).strip():
                kept[-1] = line
                continue
        kept.append(line)
    return "\n".join(kept)


def parse_metrics_from_log(log_text: str) -> Dict[str, Optional[float]]:
    """抓 [none] 标记对应的最后一条 P/R/F1。"""
    out = {"P": None, "R": None, "F1": None}
    hits = re.findall(r"\[none\]\s*P=([\d.]+),\s*R=([\d.]+),\s*F1=([\d.]+)", log_text)
    if hits:
        p, r, f = hits[-1]
        out["P"], out["R"], out["F1"] = float(p), float(r), float(f)
    return out


def parse_progress_from_log(log_text: str) -> Optional[int]:
    """从主流程日志里估算一个 0-100 的进度。

    - 优先看 `[phase X/Y]` 标记(自定义阶段序号)。
    - 其次看末尾一条 tqdm 的百分比,按 6 个阶段平均分配(启发式)。
    - 都没有就返回 None。
    """
    phase_hits = _PHASE_RE.findall(log_text)
    if phase_hits:
        cur, total = int(phase_hits[-1][0]), int(phase_hits[-1][1])
        return min(100, int(cur / max(total, 1) * 100))

    last_pct = None
    for line in reversed(log_text.splitlines()[-40:]):
        m = _PCT_RE.search(line)
        if m:
            last_pct = int(m.group(1))
            break
    return last_pct


def parse_phase_times_from_log(log_text: str) -> Dict[str, float]:
    """抓 `阶段名: 1.2345s` 形式的耗时。"""
    pat = re.compile(r"(\w+):\s*([\d.]+)s")
    out: Dict[str, float] = {}
    for line in log_text.splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_loss_from_log(log_text: str) -> List[Dict]:
    """抓对比学习 loss 序列。"""
    rows: List[Dict] = []
    for line in log_text.splitlines():
        lm = re.search(r"loss[=:]\s*([\d.]+)", line)
        if not lm:
            continue
        em = re.search(r"[Ee]poch\s*(\d+)", line)
        rows.append({
            "epoch": int(em.group(1)) if em else len(rows) + 1,
            "loss": float(lm.group(1)),
        })
    return rows


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def scan_datasets() -> List[str]:
    """列出 data/ 下所有合法数据集(包含 table_0.csv)。"""
    d = PROJECT_ROOT / "data"
    if not d.exists():
        return []
    return sorted(
        [p.name for p in d.iterdir()
         if p.is_dir() and (p / "table_0.csv").exists()]
    )


def _count_csv_rows(path: Path) -> int:
    """高效统计 CSV 行数(除表头)。"""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def _read_csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            return next(reader, []) or []
    except (OSError, StopIteration):
        return []


def get_dataset_stats(name: str) -> Dict:
    """返回数据集摘要:表数/总记录数/属性列/是否已有 LLM 正例对。"""
    dpath = PROJECT_ROOT / "data" / name
    stats = {
        "name": name,
        "exists": dpath.exists(),
        "tables": 0,
        "records": 0,
        "attrs": [],
        "has_pairs": False,
        "pairs_count": 0,
        "ground_truth": False,
    }
    if not stats["exists"]:
        return stats

    table_files = sorted(dpath.glob("table_*.csv"))
    stats["tables"] = len(table_files)
    if table_files:
        stats["attrs"] = _read_csv_header(table_files[0])
        # 大文件只对前 3 张表统计,避免 Person 这种 5M 的全扫
        sample = table_files[:3]
        partial = sum(_count_csv_rows(p) for p in sample)
        if len(table_files) > len(sample):
            # 按均值外推剩余表
            avg = partial / max(len(sample), 1)
            stats["records"] = int(partial + avg * (len(table_files) - len(sample)))
            stats["records_approx"] = True
        else:
            stats["records"] = partial
            stats["records_approx"] = False

    gt = dpath / "ground_truth.txt"
    stats["ground_truth"] = gt.exists()

    pairs_file = PROJECT_ROOT / "llm_training_data" / name / "labeled_pairs.json"
    if pairs_file.exists():
        stats["has_pairs"] = True
        try:
            raw = json.loads(pairs_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                meta = raw.get("metadata", {})
                stats["pairs_count"] = int(meta.get("num_pairs") or len(raw.get("entity_groups", [])))
            elif isinstance(raw, list):
                stats["pairs_count"] = len(raw)
        except (json.JSONDecodeError, OSError):
            pass

    return stats


# ---------------------------------------------------------------------------
# 数据集预设参数(从 src/data_chuli/dataset_configs.py 同步的简化版)
# 只暴露 UI 需要展示的字段,避免把 dataclass 跨模块传来传去
# ---------------------------------------------------------------------------
DATASET_PRESETS: Dict[str, Dict] = {
    "Geo":        {"col_sim_threshold": 0.8, "min_dis": 0.50, "selection_rate": 0.2, "k": 1, "note": "F1≈90.9"},
    "Music-20":   {"col_sim_threshold": 0.9, "min_dis": 0.35, "selection_rate": 0.2, "k": 1, "note": "F1≈90.2"},
    "Music-200":  {"col_sim_threshold": 0.9, "min_dis": 0.35, "selection_rate": 0.2, "k": 1, "note": "F1≈82.4"},
    "Music-2000": {"col_sim_threshold": 0.8, "min_dis": 0.30, "selection_rate": 0.2, "k": 1, "note": "待调优"},
    "Shopee":     {"col_sim_threshold": 0.9, "min_dis": 0.50, "selection_rate": 0.2, "k": 1, "note": "F1≈28.8"},
    "Person":     {"col_sim_threshold": 0.85,"min_dis": 0.40, "selection_rate": 0.2, "k": 1, "note": "大规模"},
}


def get_preset(name: str) -> Dict:
    """取数据集预设;不存在时返回通用默认。"""
    return DATASET_PRESETS.get(name, {
        "col_sim_threshold": 0.8, "min_dis": 0.5, "selection_rate": 0.2, "k": 1, "note": "-",
    })


# ---------------------------------------------------------------------------
# 结果文件
# ---------------------------------------------------------------------------
def load_result_files() -> List[Dict]:
    """读 results/*.json 汇总列表。"""
    rdir = PROJECT_ROOT / "results"
    if not rdir.exists():
        return []

    records: List[Dict] = []
    for f in sorted(rdir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        rec: Dict = {
            "file": f.name,
            "dataset": data.get("dataset", ""),
            "timestamp": data.get("timestamp", ""),
        }
        for ev in data.get("evaluations", []):
            if ev.get("is_final"):
                rec["P"] = ev.get("precision")
                rec["R"] = ev.get("recall")
                rec["F1"] = ev.get("f1")
                break
        params = data.get("parameters", {})
        rec["model_type"] = params.get("model_type", "")
        rec["min_dis"] = params.get("min_dis")
        rec["col_sim_threshold"] = params.get("col_sim_threshold")
        rec["use_smart_pairing"] = params.get("use_smart_pairing")
        rec["use_cl"] = params.get("use_contrastive_learning")

        summary = data.get("run_summary", {})
        rec["total_time"] = summary.get("total_time")
        rec["phase_times"] = summary.get("phase_times", {})
        records.append(rec)

    return records


def load_result_detail(filename: str) -> Optional[Dict]:
    fp = PROJECT_ROOT / "results" / filename
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
