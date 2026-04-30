"""LLM 数据生成 tab — 配置 + 日志 + 结果预览(entity group 展开)。"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from web.runner import ProcessRunner
from web.utils import scan_datasets, PROJECT_ROOT, collapse_progress_lines

runner = ProcessRunner()


def _status(state: str, count: int = 0) -> str:
    if state == "running":
        return '<div class="status-line on">正在调用 LLM 生成实体组 …</div>'
    if state == "done":
        return f'<div class="status-line on">生成完成 · 共 {count:,} 条正例对</div>'
    if state == "stopped":
        return '<div class="status-line warn">已停止生成</div>'
    return '<div class="status-line">就绪 · 配置 LLM 参数后点击「开始生成」</div>'


def _pairs_preview(path: Path) -> str:
    """把 labeled_pairs.json 渲染成学术排版的前 8 组实体对。"""
    if not path.exists():
        return '<div class="info-sheet">尚未找到 labeled_pairs.json</div>'
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f'<div class="info-sheet">读取失败: {e}</div>'

    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    groups = data.get("entity_groups", []) if isinstance(data, dict) else data
    if not isinstance(groups, list):
        return '<div class="info-sheet">文件结构不识别</div>'

    head = ""
    if meta:
        items = []
        for k in ("dataset", "num_entity_groups", "variants_per_entity", "num_pairs"):
            if k in meta:
                items.append(
                    f'<span class="label">{k}</span><span class="value">{meta[k]}</span>'
                )
        if items:
            head = '<div class="info-sheet">' + "".join(f"<div>{x}</div>" for x in items) + "</div>"

    rows_html = []
    for i, g in enumerate(groups[:8]):
        variants = g.get("variants") or []
        if not variants:
            continue
        header = (
            f'<tr><th colspan="2" style="background:var(--paper-sub);">'
            f'Group {g.get("group_id", i)} &nbsp; · &nbsp; '
            f'<span style="font-family:var(--mono); font-weight:400; color:var(--ink-soft);">'
            f'{" / ".join(str(g.get(k, "")) for k in ("canonical_givenname", "canonical_surname", "canonical_suburb") if g.get(k))}'
            f'</span></th></tr>'
        )
        body_rows = []
        for v in variants:
            style = v.get("style", "")
            # 提取除 style 外的字段
            fields = {k: val for k, val in v.items() if k != "style"}
            text = " &nbsp;·&nbsp; ".join(f"{k}=<code>{val}</code>" for k, val in fields.items())
            body_rows.append(
                f'<tr><td style="width:80px; color:var(--ink-faint); font-family:var(--mono);">{style}</td>'
                f'<td>{text}</td></tr>'
            )
        rows_html.append(
            '<table style="width:100%; border-collapse:collapse; margin-bottom:12px; '
            'font-size:0.86em; border:1px solid var(--rule);">'
            + header + "".join(body_rows) + "</table>"
        )

    if not rows_html:
        return head + '<div class="info-sheet">无实体组可预览</div>'

    return head + '<div style="margin-top:10px;">' + "".join(rows_html) + "</div>"


def run_llm_gen(
    dataset, backend, api_url, api_key, model,
    num_entities, batch_size, temperature, max_workers,
):
    params = dict(
        dataset=dataset, data_dir="data",
        backend=backend, api_url=api_url, api_key=api_key, model=model,
        num_entities=int(num_entities), batch_size=int(batch_size),
        temperature=temperature, max_workers=int(max_workers),
    )

    log_text = ""
    for line in runner.run_llm_gen(**params):
        log_text += line
        yield (
            collapse_progress_lines(log_text),
            '<div class="info-sheet">生成中 · 完成后自动预览</div>',
            _status("running"),
        )

    output_path = PROJECT_ROOT / "llm_training_data" / dataset / "labeled_pairs.json"
    preview_html = _pairs_preview(output_path)
    # 提取总数用于状态行
    count = 0
    if output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                count = int(data.get("metadata", {}).get("num_pairs") or 0)
            elif isinstance(data, list):
                count = len(data)
        except (json.JSONDecodeError, OSError):
            pass

    yield (
        collapse_progress_lines(log_text),
        preview_html,
        _status("done", count),
    )


def stop_llm_gen():
    runner.stop()
    return _status("stopped")


def create_tab():
    datasets = scan_datasets() or ["Geo"]

    gr.HTML("""
    <div class="eyebrow">Chapter III &nbsp;·&nbsp; Synthesis</div>
    <div class="section-h"><span class="idx">§ 01</span>LLM 合成训练对</div>
    <p class="section-desc dropcap">
      调用大模型分析多表异构模式,一次性生成固定数量的领域正例对,保存为
      <code>labeled_pairs.json</code> 供对比学习阶段使用。
      生成完成后,下方将预览前 8 组实体及其跨表变体。
    </p>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            gr.HTML('<div class="section-h"><span class="idx">§ 1.1</span>目标数据集</div>')
            dataset = gr.Dropdown(choices=datasets, value=datasets[0], label="Dataset", show_label=False)

            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 1.2</span>后端</div>')
            backend = gr.Radio(
                choices=["api", "ollama", "vllm"], value="api",
                label="后端类型",
                info="api: OpenAI 兼容; ollama/vllm: 本地推理",
            )
            api_url = gr.Textbox(value="https://api520.pro", label="API URL")
            api_key = gr.Textbox(value="", label="API Key", type="password")
            model = gr.Textbox(value="gpt-4o-mini", label="模型名")

            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 1.3</span>生成设置</div>')
            num_entities = gr.Slider(10, 1000, value=200, step=10, label="目标实体组数 N")
            with gr.Row():
                batch_size = gr.Number(value=4, precision=0, label="batch")
                max_workers = gr.Slider(1, 20, value=4, step=1, label="并发")
            temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="温度 T", info="越高越多样")

            with gr.Row(elem_classes="preset-row"):
                run_btn = gr.Button("开始生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">§ 02</span>运行状态</div>')
            status_html = gr.HTML(_status("idle"))

            gr.HTML('<div class="section-h" style="margin-top:14px;"><span class="idx">§ 03</span>日志</div>')
            log_output = gr.Textbox(
                show_label=False, label="",
                lines=14, max_lines=36, interactive=False, autoscroll=True,
                elem_classes="log-box",
                placeholder="生成日志 …",
            )

            gr.HTML('<div class="section-h" style="margin-top:18px;"><span class="idx">§ 04</span>实体组预览<span style="font-family:var(--serif); font-style:italic; font-weight:400; font-size:0.72em; color:var(--ink-faint); letter-spacing:0; text-transform:none; margin-left:8px;">前 8 组</span></div>')
            preview_html = gr.HTML(
                '<div class="info-sheet">生成完成后,此处会展示前 8 组合成实体的各表变体。</div>'
            )

    run_btn.click(
        fn=run_llm_gen,
        inputs=[dataset, backend, api_url, api_key, model,
                num_entities, batch_size, temperature, max_workers],
        outputs=[log_output, preview_html, status_html],
    )
    stop_btn.click(fn=stop_llm_gen, outputs=status_html)
