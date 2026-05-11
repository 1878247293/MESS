"""对比学习 tab：超参 + loss 曲线 + 终端日志"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from web.runner import ProcessRunner
from web.utils import scan_datasets, parse_loss_from_log, collapse_progress_lines

runner = ProcessRunner()


def _status(running: bool, cur: int = 0, total: int = 0, msg: str = "") -> str:
    if running:
        pct = min(100, int(cur / max(total, 1) * 100)) if total else 0
        bar = f'<div class="progress-track"><div class="progress-bar" style="width:{pct}%"></div></div>'
        return (
            f'<div class="status-line on">'
            f'训练中 · epoch {cur}/{total} · {pct}%'
            f'{bar}</div>'
        )
    if "停止" in msg:
        return f'<div class="status-line warn">{msg}</div>'
    if msg:
        return f'<div class="status-line">{msg}</div>'
    return '<div class="status-line">就绪 · 配置完点开始训练</div>'


def run_contrastive(
    data_name, model_type, cl_mode, cl_epochs, cl_batch_size,
    cl_learning_rate, cl_temperature, force_retrain,
):
    params = dict(
        data_name=data_name, model_type=model_type,
        cl_mode=cl_mode,
        cl_epochs=int(cl_epochs), cl_batch_size=int(cl_batch_size),
        cl_learning_rate=cl_learning_rate, cl_temperature=cl_temperature,
        force_retrain=force_retrain,
        use_contrastive_learning=True,
    )

    log_text = ""
    total = int(cl_epochs)
    for line in runner.run_contrastive(**params):
        log_text += line
        display = collapse_progress_lines(log_text)
        losses = parse_loss_from_log(log_text)
        df = pd.DataFrame(losses) if losses else pd.DataFrame({"epoch": [], "loss": []})
        yield display, df, _status(True, len(losses), total)

    losses = parse_loss_from_log(log_text)
    df = pd.DataFrame(losses) if losses else pd.DataFrame({"epoch": [], "loss": []})
    yield (
        collapse_progress_lines(log_text),
        df,
        _status(False, len(losses), total, f"训练结束 · 共 {len(losses)} 个 loss 点"),
    )


def stop_contrastive():
    runner.stop()
    return _status(False, 0, 0, "已停止训练")


def create_tab():
    datasets = scan_datasets() or ["Geo"]

    gr.HTML("""
    <div class="eyebrow">Chapter II &nbsp;·&nbsp; Contrastive</div>
    <div class="section-h"><span class="idx">§ 01</span>对比学习微调</div>
    <p class="section-desc dropcap">
      基于 InfoNCE，用 LLM 合成的正例对或真实标签微调底层 S-BERT 编码器。
      左边设超参，右边看 loss 曲线和训练日志。
    </p>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            gr.HTML('<div class="section-h"><span class="idx">§ 1.1</span>数据集与模型</div>')
            data_name = gr.Dropdown(choices=datasets, value=datasets[0], label="数据集")
            model_type = gr.Dropdown(
                choices=["modernbert", "minilm"], value="modernbert",
                label="骨干模型",
            )

            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 1.2</span>超参数</div>')
            cl_mode = gr.Radio(
                choices=["self-supervised", "supervised"],
                value="self-supervised", label="训练模式",
                info="self-supervised：只用合成正例；supervised：用真值标签",
            )
            with gr.Row():
                cl_epochs = gr.Slider(1, 100, value=10, step=1, label="epochs")
                cl_batch_size = gr.Number(value=64, precision=0, label="batch_size")
            with gr.Row():
                cl_learning_rate = gr.Number(value=1e-5, label="学习率 η")
                cl_temperature = gr.Slider(0.01, 1.0, value=0.07, step=0.01, label="温度 τ")

            with gr.Accordion("更多", open=False):
                force_retrain = gr.Checkbox(value=False, label="忽略缓存，强制重训")

            with gr.Row(elem_classes="preset-row"):
                run_btn = gr.Button("开始训练", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">§ 02</span>训练监控</div>')
            status_html = gr.HTML(_status(False))
            loss_plot = gr.LinePlot(
                x="epoch", y="loss",
                title="",
                x_title="epoch", y_title="InfoNCE loss",
                height=260, show_label=False,
            )
            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 03</span>日志</div>')
            log_output = gr.Textbox(
                show_label=False, label="",
                lines=18, max_lines=40,
                interactive=False, autoscroll=True,
                elem_classes="log-box",
                placeholder="训练日志 …",
            )

    run_btn.click(
        fn=run_contrastive,
        inputs=[
            data_name, model_type, cl_mode, cl_epochs, cl_batch_size,
            cl_learning_rate, cl_temperature, force_retrain,
        ],
        outputs=[log_output, loss_plot, status_html],
    )
    stop_btn.click(fn=stop_contrastive, outputs=status_html)
