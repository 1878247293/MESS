"""结果 tab：汇总表 + 单行详情 + 多选对比 + 按数据集分组"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import gradio as gr
import pandas as pd

from web.utils import load_result_files, load_result_detail


# 汇总
def _summary_card(records: List[Dict]) -> str:
    if not records:
        return """
        <div class="info-sheet">
          还没有结果文件。先在主流程 tab 跑一次实验，然后回这里刷新。
        </div>
        """
    cnt = len(records)
    f1s = [r["F1"] for r in records if r.get("F1") is not None]
    best_f1 = max(f1s) if f1s else 0.0
    best_run = next((r for r in records if r.get("F1") == best_f1), None)
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    by_ds = defaultdict(list)
    for r in records:
        if r.get("F1") is not None:
            by_ds[r.get("dataset") or "-"].append(r["F1"])
    group_rows = []
    for ds, arr in sorted(by_ds.items()):
        arr_sorted = sorted(arr, reverse=True)
        group_rows.append(
            f'<div><span class="label">{ds}</span>'
            f'<span class="value">{len(arr)} 次</span>'
            f'<span style="margin-left:16px; color:var(--ink-faint); font-family:var(--mono);">'
            f'best {arr_sorted[0]:.4f} · avg {sum(arr)/len(arr):.4f}</span>'
            f'</div>'
        )

    return f"""
    <div class="metric-grid">
      <div class="metric-cell">
        <div class="k">Total runs</div>
        <div class="v">{cnt}</div>
      </div>
      <div class="metric-cell">
        <div class="k">Best F1</div>
        <div class="v">{best_f1:.4f}</div>
        <div class="note">{best_run.get('dataset', '') if best_run else ''} · {best_run.get('file', '')[:24] if best_run else ''}</div>
      </div>
      <div class="metric-cell">
        <div class="k">Average F1</div>
        <div class="v">{avg_f1:.4f}</div>
        <div class="note">across {len(f1s)} scored runs</div>
      </div>
    </div>
    <div class="info-sheet">
      <div style="color:var(--ink); font-family:var(--serif); font-weight:600; margin-bottom:6px;">按数据集分组</div>
      {''.join(group_rows) if group_rows else '<div>无分组</div>'}
    </div>
    """


def _records_df(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "文件": r["file"],
            "数据集": r["dataset"] or "-",
            "P": round(r["P"], 4) if r.get("P") is not None else None,
            "R": round(r["R"], 4) if r.get("R") is not None else None,
            "F1": round(r["F1"], 4) if r.get("F1") is not None else None,
            "模型": r.get("model_type") or "-",
            "γ": r.get("col_sim_threshold"),
            "ε": r.get("min_dis"),
            "SAM": "✓" if r.get("use_smart_pairing") else "",
            "CL": "✓" if r.get("use_cl") else "",
            "耗时(s)": round(r.get("total_time") or 0, 2),
        })
    return pd.DataFrame(rows)


def refresh_results():
    records = load_result_files()
    df = _records_df(records) if records else pd.DataFrame()
    files = [r["file"] for r in records]
    return df, gr.update(choices=files, value=[]), _summary_card(records)


def on_row_select(evt: gr.SelectData, df: pd.DataFrame):
    """点表格行 -> 直接出 JSON 详情"""
    if df is None or df.empty or evt is None:
        return "没有结果或没选中行"
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        filename = df.iloc[row_idx]["文件"]
    except (IndexError, KeyError, TypeError):
        return "定位行失败"
    data = load_result_detail(filename)
    if not data:
        return f"读取 {filename} 失败"
    return json.dumps(data, indent=2, ensure_ascii=False)


def compare_selected(selected_files):
    if not selected_files:
        return pd.DataFrame(), pd.DataFrame()

    records = load_result_files()
    sel = [r for r in records if r["file"] in selected_files]

    bars = []
    for r in sel:
        label = f"{r['dataset']}·{r['file'][:14]}"
        for metric in ("P", "R", "F1"):
            bars.append({
                "运行": label,
                "指标": metric,
                "值": r.get(metric) if r.get(metric) is not None else 0.0,
            })

    param_rows = []
    for r in sel:
        param_rows.append({
            "文件": r["file"][:28],
            "数据集": r["dataset"],
            "模型": r.get("model_type") or "-",
            "γ": r.get("col_sim_threshold"),
            "ε": r.get("min_dis"),
            "SAM": "✓" if r.get("use_smart_pairing") else "",
            "CL": "✓" if r.get("use_cl") else "",
            "F1": round(r["F1"], 4) if r.get("F1") is not None else "-",
        })

    return pd.DataFrame(bars), pd.DataFrame(param_rows)


def create_tab():
    gr.HTML("""
    <div class="eyebrow">Chapter IV &nbsp;·&nbsp; Archive</div>
    <div class="section-h"><span class="idx">§ 01</span>实验记录</div>
    <p class="section-desc dropcap">
      自动扫 <code>results/*.json</code>。点表格任一行就能在下方看那次的完整 JSON；
      多选后按对比，可以画 P/R/F1 柱状图，并把参数差异列在一起。
    </p>
    """)

    summary_html = gr.HTML(_summary_card([]))

    with gr.Row(elem_classes="preset-row"):
        refresh_btn = gr.Button("刷新列表", variant="primary")

    # 表格内部状态：当前 DataFrame，行选回调要从里面捞文件名
    records_table = gr.Dataframe(
        value=pd.DataFrame(),
        label="",
        show_label=False,
        interactive=False,
        wrap=True,
        row_count=(10, "dynamic"),
    )

    gr.HTML('<div class="section-h" style="margin-top:26px;"><span class="idx">§ 02</span>运行详情<span style="font-family:var(--serif); font-style:italic; font-weight:400; font-size:0.72em; color:var(--ink-faint); letter-spacing:0; text-transform:none; margin-left:8px;">点击上表任一行</span></div>')
    detail_output = gr.Code(
        label="",
        show_label=False,
        language="json",
        lines=18,
    )

    gr.HTML('<div class="section-h" style="margin-top:26px;"><span class="idx">§ 03</span>多次运行对比</div>')
    with gr.Row():
        selected_files = gr.Dropdown(
            choices=[], multiselect=True, label="选若干文件",
            info="先刷新上面的表，再回这里多选",
        )
        compare_btn = gr.Button("对比", variant="secondary")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            compare_chart = gr.BarPlot(
                x="运行", y="值", color="指标",
                title="",
                x_title="", y_title="",
                height=280, show_label=False,
            )
        with gr.Column(scale=1):
            param_table = gr.Dataframe(
                label="",
                show_label=False,
                interactive=False,
                wrap=True,
            )

    # 事件
    refresh_btn.click(
        fn=refresh_results,
        outputs=[records_table, selected_files, summary_html],
    )
    records_table.select(
        fn=on_row_select,
        inputs=records_table,
        outputs=detail_output,
    )
    compare_btn.click(
        fn=compare_selected, inputs=selected_files,
        outputs=[compare_chart, param_table],
    )
