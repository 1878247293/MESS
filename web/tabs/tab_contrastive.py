"""Contrastive learning tab."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from web.runner import ProcessRunner
from web.utils import (
    collapse_progress_lines,
    parse_loss_from_log,
    parse_total_steps_from_log,
    scan_datasets,
)

runner = ProcessRunner()


def _status(running: bool, cur: int = 0, total: int = 0, msg: str = "") -> str:
    if running:
        pct = min(100, int(cur / max(total, 1) * 100)) if total else 0
        bar = f'<div class="progress-track"><div class="progress-bar" style="width:{pct}%"></div></div>'
        return f'<div class="status-line on">训练中 · 批次 {cur}{bar}</div>'
    if "停止" in msg:
        return f'<div class="status-line warn">{msg}</div>'
    return f'<div class="status-line">{msg or "等待开始。设置训练参数后即可执行对比学习。"}</div>'


def run_contrastive(
    data_name,
    model_type,
    cl_epochs,
    cl_batch_size,
    cl_learning_rate,
    cl_temperature,
    force_retrain,
):
    params = dict(
        data_name=data_name,
        model_type=model_type,
        cl_epochs=int(cl_epochs),
        cl_batch_size=int(cl_batch_size),
        cl_learning_rate=cl_learning_rate,
        cl_temperature=cl_temperature,
        force_retrain=force_retrain,
        use_contrastive_learning=True,
    )

    log_text = ""
    for line in runner.run_contrastive(**params):
        log_text += line
        losses = parse_loss_from_log(log_text)
        total = parse_total_steps_from_log(log_text)
        df = pd.DataFrame(losses) if losses else pd.DataFrame({"batch": [], "loss": []})
        yield collapse_progress_lines(log_text), df, _status(True, len(losses), total)

    losses = parse_loss_from_log(log_text)
    total = parse_total_steps_from_log(log_text)
    df = pd.DataFrame(losses) if losses else pd.DataFrame({"batch": [], "loss": []})
    yield collapse_progress_lines(log_text), df, _status(False, len(losses), total, f"训练完成，共记录 {len(losses)} 个 batch loss 点。")


def stop_contrastive():
    runner.stop()
    return _status(False, 0, 0, "已停止训练。")


def create_tab():
    datasets = scan_datasets() or ["Geo"]

    gr.HTML(
        """
        <section class="workspace">
          <div class="page-intro">
            <div class="page-card">
              <span class="section-kicker">Contrastive Learning</span>
              <h2 class="section-title">对比学习训练台</h2>
              <p class="section-desc">
                使用已有正例对或真实标签微调底层编码器。左侧配置训练超参数，右侧持续展示
                <strong>loss 曲线</strong> 与 <strong>训练日志</strong>。
              </p>
            </div>
            <div class="page-card compact">
              <span class="section-kicker">Training Focus</span>
              <p>适合快速比较不同 backbone、学习率和温度参数。</p>
              <p>当缓存模型不可靠时，可勾选强制重训。</p>
            </div>
          </div>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            gr.HTML('<div class="section-h"><span class="idx">01</span>训练配置</div>')
            data_name = gr.Dropdown(choices=datasets, value=datasets[0], label="数据集")
            model_type = gr.Dropdown(choices=["minilm"], value="minilm", label="骨干模型")
            with gr.Row():
                cl_epochs = gr.Slider(1, 100, value=20, step=1, label="Epochs")
                cl_batch_size = gr.Number(value=64, precision=0, label="Batch Size")
            with gr.Row():
                cl_learning_rate = gr.Number(value=1e-5, label="Learning Rate")
                cl_temperature = gr.Slider(0.01, 1.0, value=0.07, step=0.01, label="Temperature")

            with gr.Accordion("更多选项", open=False):
                force_retrain = gr.Checkbox(value=False, label="忽略缓存并强制重训")

            with gr.Row(elem_classes="preset-row"):
                run_btn = gr.Button("开始训练", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">02</span>训练监控</div>')
            status_html = gr.HTML(_status(False))
            loss_plot = gr.LinePlot(
                x="batch",
                y="loss",
                title="",
                x_title="batch",
                y_title="InfoNCE loss",
                height=220,
                show_label=False,
            )

            gr.HTML('<div class="section-h"><span class="idx">03</span>训练日志</div>')
            log_output = gr.Textbox(
                show_label=False,
                label="",
                lines=14,
                max_lines=32,
                interactive=False,
                autoscroll=True,
                elem_classes="log-box",
                placeholder="训练日志会在这里持续更新。",
            )

    run_btn.click(
        fn=run_contrastive,
        inputs=[
            data_name,
            model_type,
            cl_epochs,
            cl_batch_size,
            cl_learning_rate,
            cl_temperature,
            force_retrain,
        ],
        outputs=[log_output, loss_plot, status_html],
    )
    stop_btn.click(fn=stop_contrastive, outputs=status_html)
