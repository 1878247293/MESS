"""Results tab."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

import gradio as gr
import pandas as pd

from web.utils import load_result_detail, load_result_files


def _summary_card(records: List[Dict]) -> str:
    if not records:
        return '<div class="info-sheet">还没有结果文件。先运行主流程，再回到这里查看汇总。</div>'

    f1s = [item["F1"] for item in records if item.get("F1") is not None]
    best_f1 = max(f1s) if f1s else 0.0
    best_run = next((item for item in records if item.get("F1") == best_f1), None)
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    by_dataset = defaultdict(list)
    for item in records:
        if item.get("F1") is not None:
            by_dataset[item.get("dataset") or "-"].append(item["F1"])

    grouped = []
    for dataset, values in sorted(by_dataset.items()):
        grouped.append(
            f'<div><span class="label">{dataset}</span><span class="value">{len(values)} 次</span>'
            f'<span style="color:var(--text-faint);">best {max(values):.4f} · avg {sum(values) / len(values):.4f}</span></div>'
        )

    best_file = best_run.get("file", "") if best_run else "-"
    best_dataset = best_run.get("dataset", "") if best_run else "-"
    return f"""
    <div class="metric-grid">
      <div class="metric-cell">
        <span class="k">Total Runs</span>
        <span class="v">{len(records)}</span>
      </div>
      <div class="metric-cell">
        <span class="k">Best F1</span>
        <span class="v">{best_f1:.4f}</span>
        <div class="note">{best_dataset} · {best_file}</div>
      </div>
      <div class="metric-cell">
        <span class="k">Average F1</span>
        <span class="v">{avg_f1:.4f}</span>
        <div class="note">{len(f1s)} scored runs</div>
      </div>
    </div>
    <div class="info-sheet">
      <div><span class="label">数据集概览</span><span class="value">按 F1 汇总</span></div>
      {''.join(grouped)}
    </div>
    """


def _records_df(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in records:
        rows.append(
            {
                "文件": item["file"],
                "数据集": item["dataset"] or "-",
                "P": round(item["P"], 4) if item.get("P") is not None else None,
                "R": round(item["R"], 4) if item.get("R") is not None else None,
                "F1": round(item["F1"], 4) if item.get("F1") is not None else None,
                "模型": item.get("model_type") or "-",
                "列阈值": item.get("col_sim_threshold"),
                "距离阈值": item.get("min_dis"),
                "SAM": "是" if item.get("use_smart_pairing") else "",
                "CL": "是" if item.get("use_cl") else "",
                "耗时(s)": round(item.get("total_time") or 0, 2),
            }
        )
    return pd.DataFrame(rows)


def refresh_results():
    records = load_result_files()
    df = _records_df(records) if records else pd.DataFrame()
    files = [item["file"] for item in records]
    return df, gr.update(choices=files, value=[]), _summary_card(records)


def on_row_select(evt: gr.SelectData, df: pd.DataFrame):
    if df is None or df.empty or evt is None:
        return "没有可展示的结果。"
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        filename = df.iloc[row_idx]["文件"]
    except (IndexError, KeyError, TypeError):
        return "无法定位选中的结果。"
    data = load_result_detail(filename)
    if not data:
        return f"读取 {filename} 失败。"
    return json.dumps(data, indent=2, ensure_ascii=False)


def compare_selected(selected_files):
    if not selected_files:
        return pd.DataFrame(), pd.DataFrame()

    records = load_result_files()
    selected = [item for item in records if item["file"] in selected_files]
    selected.sort(key=lambda x: x.get("timestamp") or x.get("file", ""))

    bars = []
    for item in selected:
        ts = item.get("timestamp") or item["file"][:19]
        label = f"{item['dataset']} · {ts}"
        for metric in ("P", "R", "F1"):
            bars.append({"运行": label, "指标": metric, "值": item.get(metric) if item.get(metric) is not None else 0.0})

    params = []
    for item in selected:
        params.append(
            {
                "文件": item["file"][:28],
                "数据集": item["dataset"],
                "模型": item.get("model_type") or "-",
                "列阈值": item.get("col_sim_threshold"),
                "距离阈值": item.get("min_dis"),
                "SAM": "是" if item.get("use_smart_pairing") else "",
                "CL": "是" if item.get("use_cl") else "",
                "F1": round(item["F1"], 4) if item.get("F1") is not None else "-",
            }
        )

    return pd.DataFrame(bars), pd.DataFrame(params)


def create_tab():
    gr.HTML(
        """
        <section class="workspace">
          <div class="page-intro">
            <div class="page-card">
              <span class="section-kicker">Result Archive</span>
              <h2 class="section-title">实验结果分析</h2>
              <p class="section-desc">
                自动读取 <code>results/</code> 下的实验输出，支持 <strong>汇总</strong>、
                <strong>单次详情查看</strong> 和 <strong>多次运行对比</strong>。
              </p>
            </div>
            <div class="page-card compact">
              <span class="section-kicker">Review Flow</span>
              <p>先刷新列表，再点击表格查看单次结果。</p>
              <p>需要对比时，选择多个结果文件后生成图表。</p>
            </div>
          </div>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=300):
            gr.HTML('<div class="section-h"><span class="idx">01</span>结果概览</div>')
            summary_html = gr.HTML(_summary_card([]))

            gr.HTML('<div class="section-h"><span class="idx">03</span>结果详情</div>')
            detail_output = gr.Code(label="", show_label=False, language="json", lines=14)

        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">02</span>结果文件列表</div>')
            with gr.Row(elem_classes="preset-row"):
                refresh_btn = gr.Button("刷新结果列表", variant="primary")

            records_table = gr.Dataframe(
                value=pd.DataFrame(),
                label="",
                show_label=False,
                interactive=False,
                wrap=True,
                row_count=(8, "dynamic"),
            )

            gr.HTML('<div class="section-h"><span class="idx">04</span>多次运行对比</div>')
            with gr.Row():
                selected_files = gr.Dropdown(
                    choices=[],
                    multiselect=True,
                    label="选择结果文件",
                    info="可选多个结果进行横向对比。",
                    scale=4,
                )
                compare_btn = gr.Button("生成对比", variant="secondary", scale=1)

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    compare_chart = gr.LinePlot(
                        x="运行",
                        y="值",
                        color="指标",
                        title="",
                        x_title="",
                        y_title="",
                        height=200,
                        show_label=False,
                    )
                with gr.Column(scale=1):
                    param_table = gr.Dataframe(label="", show_label=False, interactive=False, wrap=True)

    refresh_btn.click(fn=refresh_results, outputs=[records_table, selected_files, summary_html])
    records_table.select(fn=on_row_select, inputs=records_table, outputs=detail_output)
    compare_btn.click(fn=compare_selected, inputs=selected_files, outputs=[compare_chart, param_table])
