"""主流程 tab：数据集概览 + 参数预设 + 实时日志 + 指标"""

from __future__ import annotations

import gradio as gr

from web.runner import ProcessRunner
from web.utils import (
    scan_datasets,
    parse_metrics_from_log,
    parse_progress_from_log,
    collapse_progress_lines,
    get_dataset_stats,
    get_preset,
)

runner = ProcessRunner()


# HTML 片段
def _metric_grid(p, r, f1) -> str:
    def _cell(k: str, v, cls: str) -> str:
        if v is None:
            return f'<div class="metric-cell"><div class="k">{k}</div><div class="v dim">—</div></div>'
        return (
            f'<div class="metric-cell">'
            f'<div class="k">{k}</div>'
            f'<div class="v">{v:.4f}</div>'
            f'</div>'
        )
    return (
        '<div class="metric-grid">'
        + _cell("Precision", p, "")
        + _cell("Recall", r, "")
        + _cell("F1", f1, "")
        + "</div>"
    )


def _info_sheet(name: str) -> str:
    s = get_dataset_stats(name)
    if not s["exists"]:
        return f'<div class="info-sheet">数据集 <span class="value">{name}</span> 不在 data/ 下</div>'

    records = s["records"]
    records_str = f"{records:,}" + (" (估)" if s.get("records_approx") else "")
    attrs_str = " / ".join(s["attrs"][:6]) if s["attrs"] else "-"
    if len(s["attrs"]) > 6:
        attrs_str += f" / … (+{len(s['attrs']) - 6})"

    pairs_note = (
        f'<span class="value">{s["pairs_count"]:,}</span> 条合成对'
        if s["has_pairs"] else '<span class="value">未生成</span>'
    )
    gt_note = "已提供" if s["ground_truth"] else "缺失"

    return f"""
    <div class="info-sheet">
      <div><span class="label">表数</span><span class="value">{s["tables"]}</span></div>
      <div><span class="label">记录数</span><span class="value">{records_str}</span></div>
      <div><span class="label">属性列</span><span class="value">{attrs_str}</span></div>
      <div><span class="label">LLM 预训练</span>{pairs_note}</div>
      <div><span class="label">真值</span><span class="value">{gt_note}</span></div>
    </div>
    """


def _status(running: bool, pct: int | None, msg: str = "") -> str:
    if running:
        bar = f'<div class="progress-track"><div class="progress-bar" style="width:{pct or 0}%"></div></div>'
        label = f"运行中 · {pct}%" if pct is not None else "运行中"
        return f'<div class="status-line on">{label} &nbsp;{msg}{bar}</div>'
    if msg.startswith("已停止") or "停止" in msg:
        return f'<div class="status-line warn">{msg}</div>'
    return f'<div class="status-line">{msg or "就绪 · 选择数据集后点击开始运行"}</div>'


# 回调
def on_dataset_change(name: str):
    """切数据集只刷概览卡，不动参数（避免覆盖用户手调的值）"""
    return _info_sheet(name)


def apply_preset(name: str):
    """套该数据集的推荐预设到参数控件"""
    p = get_preset(name)
    return (
        p["col_sim_threshold"],
        p["min_dis"],
        p["selection_rate"],
        p["k"],
        _status(False, None, f"已套用 {name} 预设 · {p['note']}"),
    )


def run_main_flow(
    data_name, model_type, use_smart_pairing, smart_pairing_strategy,
    col_sim_threshold, min_dis, k, selection_rate,
    run_in_parallel, use_dataset_config, use_efficient_matching,
):
    params = dict(
        data_name=data_name, model_type=model_type,
        use_smart_pairing=use_smart_pairing,
        smart_pairing_strategy=smart_pairing_strategy,
        col_sim_threshold=col_sim_threshold, min_dis=min_dis,
        k=k, selection_rate=selection_rate,
        run_in_parallel=run_in_parallel,
        use_dataset_config=use_dataset_config,
        use_efficient_matching=use_efficient_matching,
    )

    log_text = ""
    for line in runner.run_main_flow(**params):
        log_text += line
        display = collapse_progress_lines(log_text)
        m = parse_metrics_from_log(log_text)
        pct = parse_progress_from_log(log_text)
        yield (
            display,
            _metric_grid(m["P"], m["R"], m["F1"]),
            _status(True, pct),
        )

    # 跑完了再发一帧 running=False
    m = parse_metrics_from_log(log_text)
    done_msg = "已完成" if m["F1"] is not None else "已退出（没解析到指标）"
    yield (
        collapse_progress_lines(log_text),
        _metric_grid(m["P"], m["R"], m["F1"]),
        _status(False, None, done_msg),
    )


def stop_main_flow():
    runner.stop()
    return _status(False, None, "已停止运行")


# 页面构造
def create_tab():
    datasets = scan_datasets() or ["Geo"]
    initial = datasets[0]

    gr.HTML("""
    <div class="eyebrow">Chapter I &nbsp;·&nbsp; Process</div>
    <div class="section-h"><span class="idx">§ 01</span>实体匹配流程</div>
    <p class="section-desc dropcap">
      选择数据集,套用推荐预设或手动调参,实时追踪日志与匹配指标 P/R/F1。
      每一轮运行的结果会归档到 <code>results/</code>,可在「结果」一章回看。
    </p>
    """)

    with gr.Row(equal_height=False):
        # ── 左:配置区 ──
        with gr.Column(scale=4, min_width=360):
            gr.HTML('<div class="section-h"><span class="idx">§ 1.1</span>数据集</div>')
            data_name = gr.Dropdown(
                choices=datasets, value=initial, label="Dataset",
                show_label=False, container=False,
            )
            info_html = gr.HTML(_info_sheet(initial))

            # 预设按钮行
            gr.HTML('<div class="section-h" style="margin-top:18px;"><span class="idx">§ 1.2</span>推荐预设</div>')
            with gr.Row(elem_classes="preset-row"):
                preset_btn = gr.Button("套用当前数据集推荐参数", size="sm")

            # 参数
            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 1.3</span>参数</div>')

            model_type = gr.Dropdown(
                choices=["modernbert", "minilm"], value="modernbert",
                label="嵌入模型",
                info="modernbert: 精度优先 / minilm: 速度优先",
            )
            col_sim_threshold = gr.Slider(
                0, 1, value=0.8, step=0.05,
                label="属性显著性阈值 γ (col_sim_threshold)",
                info="小于该值的列被视为噪声剔除",
            )
            min_dis = gr.Slider(
                0, 1, value=0.5, step=0.05,
                label="实体对齐阈值 ε (min_dis)",
                info="HNSW 合并时的最大距离",
            )

            with gr.Accordion("高级", open=False):
                use_smart_pairing = gr.Checkbox(value=False, label="语义感知调度 (SAM)")
                smart_pairing_strategy = gr.Radio(
                    choices=["similarity", "optimal", "complementary"],
                    value="optimal", label="调度",
                )
                k = gr.Number(value=1, precision=0, label="KNN k")
                selection_rate = gr.Slider(
                    0, 1, value=0.2, step=0.05,
                    label="采样率 selection_rate",
                )
                run_in_parallel = gr.Checkbox(value=False, label="并行合并")
                use_dataset_config = gr.Checkbox(
                    value=True, label="启动时套用数据集最优配置（会盖掉上面的参数）",
                )
                use_efficient_matching = gr.Checkbox(value=False, label="efficient 匹配")

            with gr.Row(elem_classes="preset-row"):
                run_btn = gr.Button("开始运行", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        # ── 右:输出区 ──
        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">§ 02</span>运行状态</div>')
            status_html = gr.HTML(_status(False, None))
            metric_html = gr.HTML(_metric_grid(None, None, None))

            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 03</span>日志</div>')
            log_output = gr.Textbox(
                label="",
                show_label=False,
                lines=26, max_lines=60,
                interactive=False, autoscroll=True,
                elem_classes="log-box",
                placeholder="运行开始后,日志流将实时追加于此 …",
            )

    # ── 事件 ──
    data_name.change(fn=on_dataset_change, inputs=data_name, outputs=info_html)

    preset_btn.click(
        fn=apply_preset, inputs=data_name,
        outputs=[col_sim_threshold, min_dis, selection_rate, k, status_html],
    )

    run_btn.click(
        fn=run_main_flow,
        inputs=[
            data_name, model_type, use_smart_pairing, smart_pairing_strategy,
            col_sim_threshold, min_dis, k, selection_rate,
            run_in_parallel, use_dataset_config, use_efficient_matching,
        ],
        outputs=[log_output, metric_html, status_html],
    )
    stop_btn.click(fn=stop_main_flow, outputs=status_html)
