"""前端杂活：数据集统计、日志解析、结果读取、参数预设"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# tqdm 进度条："Loading:  10%|...| 20/199 [..."
_PROGRESS_RE = re.compile(r"^(.*?)\s*\d+%\|")
# 末尾百分比："...  35%"
_PCT_RE = re.compile(r"(\d{1,3})%")
# 自定义阶段标 [phase X/Y]
_PHASE_RE = re.compile(r"\[phase\s+(\d+)/(\d+)\]", re.IGNORECASE)


# 日志整理
def collapse_progress_lines(log_text: str) -> str:
    """同一条 tqdm 连续输出，只保留最后一帧"""
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
    """抓 [none] 标记下最后一组 P/R/F1"""
    out = {"P": None, "R": None, "F1": None}
    hits = re.findall(r"\[none\]\s*P=([\d.]+),\s*R=([\d.]+),\s*F1=([\d.]+)", log_text)
    if hits:
        p, r, f = hits[-1]
        out["P"], out["R"], out["F1"] = float(p), float(r), float(f)
    return out


def parse_progress_from_log(log_text: str) -> Optional[int]:
    """从主流程日志里挤出一个 0-100 的进度。

    优先级：[phase X/Y] > 末尾 tqdm 百分比 > None。
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
    """抓 `xxx: 1.2345s` 这种行的耗时"""
    pat = re.compile(r"(\w+):\s*([\d.]+)s")
    out: Dict[str, float] = {}
    for line in log_text.splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_loss_from_log(log_text: str) -> List[Dict]:
    """抓对比学习日志里的 loss 序列"""
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


# 数据集
def scan_datasets() -> List[str]:
    """data/ 下含 table_0.csv 的目录都算合法数据集"""
    d = PROJECT_ROOT / "data"
    if not d.exists():
        return []
    return sorted(
        [p.name for p in d.iterdir()
         if p.is_dir() and (p / "table_0.csv").exists()]
    )


def _count_csv_rows(path: Path) -> int:
    """数行数（去掉表头）"""
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
    """返回数据集摘要：表数、总行数、属性列、是否已有标注正对"""
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
        # 大表全扫太慢，前 3 张采样估算
        sample = table_files[:3]
        partial = sum(_count_csv_rows(p) for p in sample)
        if len(table_files) > len(sample):
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


# 数据集预设
# src/data_chuli/dataset_configs.py 那份的简化镜像，只挑 UI 要显示的字段
DATASET_PRESETS: Dict[str, Dict] = {
    "Geo":        {"col_sim_threshold": 0.8, "min_dis": 0.50, "selection_rate": 0.2, "k": 1, "note": "F1≈90.9"},
    "Music-20":   {"col_sim_threshold": 0.9, "min_dis": 0.35, "selection_rate": 0.2, "k": 1, "note": "F1≈90.2"},
    "Music-200":  {"col_sim_threshold": 0.9, "min_dis": 0.35, "selection_rate": 0.2, "k": 1, "note": "F1≈82.4"},
    "Music-2000": {"col_sim_threshold": 0.8, "min_dis": 0.30, "selection_rate": 0.2, "k": 1, "note": "待调"},
    "Shopee":     {"col_sim_threshold": 0.9, "min_dis": 0.50, "selection_rate": 0.2, "k": 1, "note": "F1≈28.8"},
    "Person":     {"col_sim_threshold": 0.85,"min_dis": 0.40, "selection_rate": 0.2, "k": 1, "note": "大规模"},
}


def get_preset(name: str) -> Dict:
    return DATASET_PRESETS.get(name, {
        "col_sim_threshold": 0.8, "min_dis": 0.5, "selection_rate": 0.2, "k": 1, "note": "-",
    })


# 结果文件
def load_result_files() -> List[Dict]:
    """results/*.json 汇总成列表"""
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
