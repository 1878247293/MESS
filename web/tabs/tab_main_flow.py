"""Main workflow tab."""

from __future__ import annotations

import gradio as gr

from web.runner import ProcessRunner
from web.utils import (
    collapse_progress_lines,
    get_dataset_stats,
    get_preset,
    parse_metrics_from_log,
    parse_progress_from_log,
    scan_datasets,
)

runner = ProcessRunner()


def _metric_grid(p, r, f1) -> str:
    def _cell(name: str, value) -> str:
        if value is None:
            return f'<div class="metric-cell"><span class="k">{name}</span><span class="v dim">--</span></div>'
        return f'<div class="metric-cell"><span class="k">{name}</span><span class="v">{value:.4f}</span></div>'

    return '<div class="metric-grid">' + _cell("Precision", p) + _cell("Recall", r) + _cell("F1", f1) + "</div>"


def _info_sheet(name: str) -> str:
    stats = get_dataset_stats(name)
    if not stats["exists"]:
        return f'<div class="info-sheet">数据集 <span class="value">{name}</span> 不存在于 <code>data/</code> 下。</div>'

    attrs = " / ".join(stats["attrs"][:6]) if stats["attrs"] else "-"
    if len(stats["attrs"]) > 6:
        attrs += f" / ... (+{len(stats['attrs']) - 6})"

    records = f"{stats['records']:,}"
    if stats.get("records_approx"):
        records += " (估算)"

    pair_state = f'{stats["pairs_count"]:,} 组' if stats["has_pairs"] else "未生成"
    gt_state = "已提供" if stats["ground_truth"] else "缺失"

    return f"""
    <div class="info-sheet">
      <div><span class="label">表数量</span><span class="value">{stats["tables"]}</span></div>
      <div><span class="label">记录数</span><span class="value">{records}</span></div>
      <div><span class="label">字段</span><span class="value">{attrs}</span></div>
      <div><span class="label">LLM 正例</span><span class="value">{pair_state}</span></div>
      <div><span class="label">Ground Truth</span><span class="value">{gt_state}</span></div>
    </div>
    """


def _status(running: bool, pct: int | None, msg: str = "") -> str:
    if running:
        bar = f'<div class="progress-track"><div class="progress-bar" style="width:{pct or 0}%"></div></div>'
        label = f"运行中 · {pct}%" if pct is not None else "运行中"
        return f'<div class="status-line on">{label}<br>{msg or "正在执行主流程"}{bar}</div>'
    if "停止" in msg:
        return f'<div class="status-line warn">{msg}</div>'
    return f'<div class="status-line">{msg or "等待开始。选择数据集并确认参数后执行主流程。"}</div>'


def on_dataset_change(name: str):
    return _info_sheet(name)


def apply_preset(name: str):
    preset = get_preset(name)
    return (
        preset["col_sim_threshold"],
        preset["min_dis"],
        preset["selection_rate"],
        preset["k"],
        _status(False, None, f"已应用 {name} 推荐参数。{preset['note']}"),
    )


def run_main_flow(
    data_name,
    model_type,
    use_smart_pairing,
    smart_pairing_strategy,
    col_sim_threshold,
    min_dis,
    k,
    selection_rate,
    run_in_parallel,
    use_dataset_config,
    use_efficient_matching,
):
    params = dict(
        data_name=data_name,
        model_type=model_type,
        use_smart_pairing=use_smart_pairing,
        smart_pairing_strategy=smart_pairing_strategy,
        col_sim_threshold=col_sim_threshold,
        min_dis=min_dis,
        k=k,
        selection_rate=selection_rate,
        run_in_parallel=run_in_parallel,
        use_dataset_config=use_dataset_config,
        use_efficient_matching=use_efficient_matching,
    )

    log_text = ""
    for line in runner.run_main_flow(**params):
        log_text += line
        metrics = parse_metrics_from_log(log_text)
        progress = parse_progress_from_log(log_text)
        yield (
            collapse_progress_lines(log_text),
            _metric_grid(metrics["P"], metrics["R"], metrics["F1"]),
            _status(True, progress),
        )

    metrics = parse_metrics_from_log(log_text)
    final_msg = "运行完成。" if metrics["F1"] is not None else "运行结束，但未解析到最终指标。"
    yield (
        collapse_progress_lines(log_text),
        _metric_grid(metrics["P"], metrics["R"], metrics["F1"]),
        _status(False, None, final_msg),
    )


def stop_main_flow():
    runner.stop()
    return _status(False, None, "已停止主流程任务。")


def create_tab():
    datasets = scan_datasets() or ["Geo"]
    initial = datasets[0]

    gr.HTML(
        """
        <section class="workspace">
          <div class="page-intro">
            <div class="page-card">
              <span class="section-kicker">Main Pipeline</span>
              <h2 class="section-title">实体匹配主流程</h2>
              <p class="section-desc">
                在这里完成 <strong>数据集选择</strong>、<strong>参数设定</strong> 与 <strong>运行监控</strong>。
                页面右侧持续显示状态、日志和最终 P / R / F1，便于快速迭代参数。
              </p>
            </div>
            <div class="page-card compact">
              <span class="section-kicker">How To Use</span>
              <p>1. 先确认数据集摘要。</p>
              <p>2. 可直接加载推荐参数，再微调高级选项。</p>
              <p>3. 运行结束后可去结果分析页查看历史实验。</p>
            </div>
          </div>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            gr.HTML('<div class="section-h"><span class="idx">01</span>数据集与参数</div>')
            data_name = gr.Dropdown(choices=datasets, value=initial, label="数据集", container=False)
            info_html = gr.HTML(_info_sheet(initial))

            with gr.Row(elem_classes="preset-row"):
                preset_btn = gr.Button("应用推荐参数", size="sm")

            model_type = gr.Dropdown(
                choices=["modernbert", "minilm"],
                value="modernbert",
                label="嵌入模型",
                info="modernbert 偏向精度，minilm 偏向速度。",
            )
            col_sim_threshold = gr.Slider(
                0,
                1,
                value=0.8,
                step=0.05,
                label="列相似度阈值",
                info="低于该值的列将被视为噪声并剔除。",
            )
            min_dis = gr.Slider(
                0,
                1,
                value=0.5,
                step=0.05,
                label="实体对齐距离阈值",
                info="控制 HNSW 合并时允许的最大距离。",
            )

            with gr.Accordion("高级设置", open=False):
                use_smart_pairing = gr.Checkbox(value=False, label="启用语义感知调度")
                smart_pairing_strategy = gr.Radio(
                    choices=["similarity", "optimal", "complementary"],
                    value="optimal",
                    label="调度策略",
                )
                k = gr.Number(value=1, precision=0, label="KNN k")
                selection_rate = gr.Slider(0, 1, value=0.2, step=0.05, label="采样率")
                run_in_parallel = gr.Checkbox(value=False, label="并行合并")
                use_dataset_config = gr.Checkbox(value=True, label="优先使用数据集默认配置")
                use_efficient_matching = gr.Checkbox(value=False, label="启用高效匹配")

            with gr.Row(elem_classes="preset-row"):
                run_btn = gr.Button("开始运行", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">02</span>运行监控</div>')
            status_html = gr.HTML(_status(False, None))
            metric_html = gr.HTML(_metric_grid(None, None, None))

            gr.HTML('<div class="section-h"><span class="idx">03</span>实时日志</div>')
            log_output = gr.Textbox(
                label="",
                show_label=False,
                lines=26,
                max_lines=60,
                interactive=False,
                autoscroll=True,
                elem_classes="log-box",
                placeholder="运行开始后，这里会持续输出日志。",
            )

    data_name.change(fn=on_dataset_change, inputs=data_name, outputs=info_html)
    preset_btn.click(
        fn=apply_preset,
        inputs=data_name,
        outputs=[col_sim_threshold, min_dis, selection_rate, k, status_html],
    )
    run_btn.click(
        fn=run_main_flow,
        inputs=[
            data_name,
            model_type,
            use_smart_pairing,
            smart_pairing_strategy,
            col_sim_threshold,
            min_dis,
            k,
            selection_rate,
            run_in_parallel,
            use_dataset_config,
            use_efficient_matching,
        ],
        outputs=[log_output, metric_html, status_html],
    )
    stop_btn.click(fn=stop_main_flow, outputs=status_html)
