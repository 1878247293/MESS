"""LLM data generation tab."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from web.runner import ProcessRunner
from web.utils import PROJECT_ROOT, collapse_progress_lines, scan_datasets

runner = ProcessRunner()


def _status(state: str, count: int = 0) -> str:
    if state == "running":
        return '<div class="status-line on">正在调用 LLM 生成实体组与正例对。</div>'
    if state == "done":
        return f'<div class="status-line on">生成完成，共写入 {count:,} 组样本。</div>'
    if state == "stopped":
        return '<div class="status-line warn">已停止生成任务。</div>'
    return '<div class="status-line">等待开始。填写模型信息和生成规模后即可执行。</div>'


def _pairs_preview(path: Path) -> str:
    if not path.exists():
        return '<div class="info-sheet">当前还没有 <code>labeled_pairs.json</code>。</div>'
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return f'<div class="info-sheet">读取预览失败：{exc}</div>'

    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    groups = data.get("entity_groups", []) if isinstance(data, dict) else data
    if not isinstance(groups, list):
        return '<div class="info-sheet">文件结构不符合预期。</div>'

    head = ""
    if meta:
        items = []
        for key in ("dataset", "num_entity_groups", "variants_per_entity", "num_pairs"):
            if key in meta:
                items.append(f'<div><span class="label">{key}</span><span class="value">{meta[key]}</span></div>')
        head = '<div class="info-sheet">' + "".join(items) + "</div>"

    tables = []
    for index, group in enumerate(groups[:8]):
        variants = group.get("variants") or []
        if not variants:
            continue
        canonical = " / ".join(
            str(group.get(key, ""))
            for key in ("canonical_givenname", "canonical_surname", "canonical_suburb")
            if group.get(key)
        )
        rows_html = []
        for item in variants:
            style = item.get("style", "-")
            fields = " · ".join(f"{k}=<code>{v}</code>" for k, v in item.items() if k != "style")
            rows_html.append(
                '<tr style="border-top:1px solid rgba(156,183,216,0.22);">'
                f'<td style="width:92px; padding:10px 14px; color:var(--text-faint);">{style}</td>'
                f'<td style="padding:10px 14px;">{fields}</td>'
                "</tr>"
            )
        tables.append(
            '<table style="width:100%; border-collapse:collapse; margin-top:12px; background:#fff; border:1px solid rgba(156,183,216,0.4); border-radius:12px; overflow:hidden;">'
            f'<tr><th colspan="2" style="text-align:left; padding:12px 14px; background:rgba(220,234,254,0.55); color:var(--text);">Group {group.get("group_id", index)}'
            f'<span style="margin-left:10px; color:var(--text-faint); font-weight:500;">{canonical}</span></th></tr>'
            + "".join(rows_html)
            + "</table>"
        )

    if not tables:
        return head + '<div class="info-sheet">没有可展示的实体组。</div>'

    return head + "".join(tables)


def run_llm_gen(dataset, backend, api_url, api_key, model, num_entities, batch_size, temperature, max_workers):
    params = dict(
        dataset=dataset,
        data_dir="data",
        backend=backend,
        api_url=api_url,
        api_key=api_key,
        model=model,
        num_entities=int(num_entities),
        batch_size=int(batch_size),
        temperature=temperature,
        max_workers=int(max_workers),
    )

    log_text = ""
    for line in runner.run_llm_gen(**params):
        log_text += line
        yield collapse_progress_lines(log_text), '<div class="info-sheet">生成中，完成后将自动刷新预览。</div>', _status("running")

    output_path = PROJECT_ROOT / "llm_training_data" / dataset / "labeled_pairs.json"
    preview_html = _pairs_preview(output_path)
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

    yield collapse_progress_lines(log_text), preview_html, _status("done", count)


def stop_llm_gen():
    runner.stop()
    return _status("stopped")


def create_tab():
    datasets = scan_datasets() or ["Geo"]

    gr.HTML(
        """
        <section class="workspace">
          <div class="page-intro">
            <div class="page-card">
              <span class="section-kicker">LLM Synthesis</span>
              <h2 class="section-title">训练数据生成</h2>
              <p class="section-desc">
                配置大模型后端、样本规模和生成温度，自动构建可用于训练的实体组与正例对。
                生成完成后，下方会展示前 8 组实体变体预览。
              </p>
            </div>
            <div class="page-card compact">
              <span class="section-kicker">Output</span>
              <p>默认输出到 <code>llm_training_data/&lt;dataset&gt;/labeled_pairs.json</code>。</p>
              <p>适合在主流程前先准备补充训练样本。</p>
            </div>
          </div>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=360):
            gr.HTML('<div class="section-h"><span class="idx">01</span>生成配置</div>')
            dataset = gr.Dropdown(choices=datasets, value=datasets[0], label="数据集", show_label=True)
            backend = gr.Radio(
                choices=["api", "ollama", "vllm"],
                value="api",
                label="后端类型",
                info="支持 OpenAI 兼容 API、本地 Ollama 和 vLLM。",
            )
            api_url = gr.Textbox(value="https://api520.pro", label="API URL")
            api_key = gr.Textbox(value="", label="API Key", type="password")
            model = gr.Textbox(value="gpt-4o-mini", label="模型名称")
            num_entities = gr.Slider(10, 1000, value=200, step=10, label="目标实体组数量")
            with gr.Row():
                batch_size = gr.Number(value=4, precision=0, label="Batch Size")
                max_workers = gr.Slider(1, 20, value=4, step=1, label="并发数")
            temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="Temperature")

            with gr.Row(elem_classes="preset-row"):
                run_btn = gr.Button("开始生成", variant="primary")
                stop_btn = gr.Button("停止", variant="stop")

        with gr.Column(scale=6):
            gr.HTML('<div class="section-h"><span class="idx">02</span>运行状态</div>')
            status_html = gr.HTML(_status("idle"))

            gr.HTML('<div class="section-h"><span class="idx">03</span>生成日志</div>')
            log_output = gr.Textbox(
                show_label=False,
                label="",
                lines=12,
                max_lines=28,
                interactive=False,
                autoscroll=True,
                elem_classes="log-box",
                placeholder="生成日志会在这里持续更新。",
            )

            gr.HTML('<div class="section-h"><span class="idx">04</span>实体组预览</div>')
            preview_html = gr.HTML('<div class="info-sheet">生成完成后，这里会展示样本预览。</div>')

    run_btn.click(
        fn=run_llm_gen,
        inputs=[dataset, backend, api_url, api_key, model, num_entities, batch_size, temperature, max_workers],
        outputs=[log_output, preview_html, status_html],
    )
    stop_btn.click(fn=stop_llm_gen, outputs=status_html)
