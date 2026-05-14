"""SAGEM Gradio frontend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from web.tabs import tab_contrastive, tab_llm_gen, tab_main_flow, tab_results


def ensure_supported_gradio() -> None:
    version_text = getattr(gr, "__version__", "unknown")
    major_text = version_text.split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError:
        return
    if major >= 6:
        raise RuntimeError(
            "Unsupported gradio version detected: "
            f"{version_text}. This project requires gradio>=4.44,<6. "
            "Reinstall with: pip install \"gradio>=4.44,<6\""
        )


def ensure_localhost_no_proxy() -> None:
    bypass_hosts = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [part.strip() for part in current.split(",") if part.strip()]
        for host in bypass_hosts:
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+SC:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #f4f8fc;
    --surface: #ffffff;
    --surface-soft: #edf4fb;
    --surface-muted: #e3edf8;
    --line: #c9d9ea;
    --line-strong: #9cb7d8;
    --text: #14324d;
    --text-soft: #4b6885;
    --text-faint: #6f8ba7;
    --primary: #1e64c8;
    --primary-deep: #184f9f;
    --primary-soft: #dceafe;
    --primary-ink: #ffffff;
    --success: #1f7a5c;
    --warning: #c7701d;
    --danger: #bd4f4f;
    --shadow: 0 18px 50px rgba(24, 71, 128, 0.08);
    --radius-lg: 20px;
    --radius-md: 14px;
    --radius-sm: 10px;
    --sans: "IBM Plex Sans", "IBM Plex Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    --mono: "JetBrains Mono", "Consolas", monospace;
}

html, body {
    margin: 0;
    background:
        radial-gradient(circle at top left, rgba(210, 228, 248, 0.75), transparent 36%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
    color: var(--text);
    font-family: var(--sans);
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(158, 183, 213, 0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(158, 183, 213, 0.12) 1px, transparent 1px);
    background-size: 24px 24px;
    mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.38), transparent 78%);
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    background: transparent !important;
    padding: 0 !important;
}

.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .contain,
.gradio-container .app,
.gradio-container > div {
    max-width: 100% !important;
    width: 100% !important;
    box-sizing: border-box;
}

.app-shell {
    width: min(1440px, calc(100vw - 40px));
    margin: 20px auto 28px;
    padding: 28px;
    border: 1px solid rgba(156, 183, 216, 0.55);
    border-radius: 28px;
    background: rgba(255, 255, 255, 0.84);
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
}

.hero {
    display: grid;
    grid-template-columns: minmax(0, 1.65fr) minmax(280px, 0.95fr);
    gap: 20px;
    margin-bottom: 20px;
}

.hero-panel,
.hero-side,
.page-card,
.panel-muted,
.status-line,
.metric-cell,
.info-sheet,
.footer-shell {
    border: 1px solid rgba(156, 183, 216, 0.48);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 249, 255, 0.96));
    box-shadow: 0 8px 24px rgba(26, 70, 121, 0.05);
}

.hero-panel {
    padding: 26px 30px;
    border-radius: 24px;
}

.hero-kicker,
.section-kicker,
.chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--primary-soft);
    color: var(--primary);
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.hero-title {
    margin: 16px 0 10px;
    font-size: clamp(28px, 3.3vw, 48px);
    line-height: 1.06;
    letter-spacing: 0;
    font-weight: 700;
    color: var(--text);
}

.hero-title .accent {
    color: var(--primary);
}

.hero-copy {
    max-width: 52rem;
    margin: 0;
    font-size: 15px;
    line-height: 1.75;
    color: var(--text-soft);
}

.hero-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 24px;
}

.hero-metrics .hero-stat {
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(220, 234, 254, 0.48);
    border: 1px solid rgba(156, 183, 216, 0.42);
}

.hero-stat .num {
    display: block;
    font-size: 28px;
    line-height: 1;
    font-weight: 700;
    color: var(--primary);
}

.hero-stat .label {
    display: block;
    margin-top: 6px;
    font-size: 12px;
    color: var(--text-faint);
}

.hero-side {
    display: grid;
    gap: 14px;
    padding: 24px;
    border-radius: 24px;
}

.hero-side h3,
.page-card h3 {
    margin: 0;
    font-size: 17px;
    line-height: 1.3;
}

.hero-side p,
.page-card p,
.section-desc {
    margin: 0;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-soft);
}

.grid-note {
    display: grid;
    gap: 10px;
}

.tab-nav {
    gap: 10px !important;
    margin: 0 0 18px !important;
    padding: 8px !important;
    border-radius: 18px !important;
    border: 1px solid rgba(156, 183, 216, 0.48) !important;
    background: rgba(235, 244, 253, 0.75) !important;
}

.tab-nav button {
    min-height: 56px !important;
    padding: 10px 18px !important;
    border-radius: 14px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: var(--text-soft) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease, transform 0.18s ease !important;
}

.tab-nav button:hover {
    background: rgba(255, 255, 255, 0.78) !important;
    color: var(--text) !important;
    transform: translateY(-1px);
}

.tab-nav button.selected {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(226, 239, 252, 0.96)) !important;
    color: var(--primary) !important;
    border-color: rgba(156, 183, 216, 0.65) !important;
}

.tab-nav button span {
    font-size: 14px !important;
    font-weight: 600 !important;
}

.workspace {
    padding: 6px 0 10px;
}

.page-intro {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.8fr);
    gap: 16px;
    margin-bottom: 20px;
}

.page-card {
    padding: 22px 24px;
    border-radius: 20px;
}

.page-card.compact {
    display: grid;
    align-content: start;
    gap: 10px;
}

.section-title {
    margin: 12px 0 8px;
    font-size: 30px;
    line-height: 1.12;
    font-weight: 700;
    letter-spacing: 0;
    color: var(--text);
}

.section-desc strong {
    color: var(--primary);
    font-weight: 600;
}

.section-h {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 12px;
    font-size: 18px;
    line-height: 1.3;
    font-weight: 700;
    color: var(--text);
}

.section-h .idx {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 10px;
    background: var(--primary-soft);
    color: var(--primary);
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
}

.panel-muted {
    padding: 12px 14px;
    border-radius: 14px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 12px 0 18px;
}

.metric-cell {
    padding: 18px;
    border-radius: 16px;
}

.metric-cell .k {
    display: block;
    font-size: 12px;
    color: var(--text-faint);
    font-family: var(--mono);
    text-transform: uppercase;
}

.metric-cell .v {
    display: block;
    margin-top: 8px;
    font-size: 32px;
    line-height: 1;
    font-weight: 700;
    color: var(--primary);
}

.metric-cell .v.dim {
    color: var(--text-faint);
}

.metric-cell .note {
    margin-top: 8px;
    font-size: 12px;
    color: var(--text-faint);
}

.info-sheet {
    display: grid;
    gap: 10px;
    padding: 16px 18px;
    border-radius: 16px;
    color: var(--text-soft);
}

.info-sheet > div {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 10px;
}

.info-sheet .label {
    min-width: 84px;
    color: var(--text-faint);
    font-size: 12px;
    font-family: var(--mono);
    text-transform: uppercase;
}

.info-sheet .value {
    color: var(--text);
    font-weight: 600;
}

.status-line {
    padding: 16px 18px;
    border-radius: 16px;
    color: var(--text);
    font-size: 14px;
}

.status-line.on {
    border-color: rgba(31, 122, 92, 0.24);
}

.status-line.warn {
    border-color: rgba(199, 112, 29, 0.26);
    color: #9b5a18;
}

.progress-track {
    width: 100%;
    height: 8px;
    margin-top: 12px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(201, 217, 234, 0.7);
}

.progress-bar {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--primary), #49a1ff);
}

.log-box textarea,
.gradio-textbox textarea,
.gradio-code textarea {
    font-family: var(--mono) !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}

.gr-form, .gr-box, .block, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

label, .gradio-container label {
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text-soft) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

.gradio-container .info,
.gradio-container label + * small {
    color: var(--text-faint) !important;
    font-size: 12px !important;
}

input, textarea, select,
.gr-box, .gr-input, .gr-dropdown,
.gradio-textbox textarea,
.gradio-textbox input,
.gradio-dropdown .wrap,
.gradio-number input {
    border-radius: 14px !important;
    border: 1px solid rgba(156, 183, 216, 0.62) !important;
    background: rgba(255, 255, 255, 0.92) !important;
    color: var(--text) !important;
    box-shadow: none !important;
}

input:focus, textarea:focus,
.gradio-dropdown .wrap:focus-within,
.gradio-number input:focus {
    border-color: var(--primary) !important;
}

.gradio-slider input[type="range"] {
    accent-color: var(--primary);
}

.gr-accordion, details {
    border: 1px solid rgba(156, 183, 216, 0.48) !important;
    border-radius: 16px !important;
    background: rgba(245, 250, 255, 0.92) !important;
    overflow: hidden !important;
}

.gr-accordion > .label-wrap, summary {
    padding: 12px 14px !important;
    color: var(--primary) !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    background: transparent !important;
}

button.primary, button[variant="primary"], .gr-button-primary {
    border-radius: 14px !important;
    border: 1px solid transparent !important;
    background: linear-gradient(180deg, #2b79e8 0%, var(--primary) 100%) !important;
    color: var(--primary-ink) !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 18px !important;
    box-shadow: 0 12px 26px rgba(30, 100, 200, 0.22) !important;
}

button.secondary, .gr-button-secondary,
button.stop, .gr-button-stop,
.preset-row button {
    border-radius: 14px !important;
    font-family: var(--sans) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 18px !important;
    box-shadow: none !important;
}

button.secondary, .gr-button-secondary,
.preset-row button {
    border: 1px solid rgba(156, 183, 216, 0.62) !important;
    background: rgba(255, 255, 255, 0.92) !important;
    color: var(--primary) !important;
}

button.stop, .gr-button-stop {
    border: 1px solid rgba(189, 79, 79, 0.34) !important;
    background: rgba(255, 245, 245, 0.98) !important;
    color: var(--danger) !important;
}

.preset-row {
    gap: 10px !important;
    margin-top: 8px !important;
}

.footer-shell {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-top: 16px;
    padding: 18px 22px;
    border-radius: 20px;
}

.footer-shell p {
    margin: 0;
    color: var(--text-soft);
    font-size: 13px;
    line-height: 1.7;
}

.mono {
    font-family: var(--mono);
}

code {
    padding: 2px 6px;
    border-radius: 8px;
    background: rgba(220, 234, 254, 0.7);
    color: var(--primary-deep);
    font-family: var(--mono);
}

@media (max-width: 1100px) {
    .hero,
    .page-intro {
        grid-template-columns: 1fr;
    }

    .hero-metrics,
    .metric-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .app-shell {
        width: min(100vw - 16px, 100%);
        margin: 8px auto 16px;
        padding: 14px;
        border-radius: 20px;
    }

    .hero-panel,
    .hero-side,
    .page-card,
    .footer-shell {
        padding: 18px;
        border-radius: 18px;
    }

    .section-title {
        font-size: 24px;
    }

    .footer-shell {
        flex-direction: column;
    }
}
"""


HERO_HTML = """
<div class="app-shell">
  <section class="hero">
    <div class="hero-panel">
      <span class="hero-kicker">SAGEM Workbench</span>
      <h1 class="hero-title">面向 <span class="accent">多表实体匹配</span> 的统一实验前端</h1>
      <p class="hero-copy">
        这是一套围绕主流程、对比学习、LLM 数据生成与结果回看构建的蓝白工作台界面。
        页面重点放在配置清晰度、运行反馈和结果可读性，避免装饰性干扰。
      </p>
      <div class="hero-metrics">
        <div class="hero-stat">
          <span class="num">4</span>
          <span class="label">核心工作区</span>
        </div>
        <div class="hero-stat">
          <span class="num">1</span>
          <span class="label">统一视觉系统</span>
        </div>
        <div class="hero-stat">
          <span class="num">实时</span>
          <span class="label">日志与状态反馈</span>
        </div>
      </div>
    </div>
    <aside class="hero-side">
      <div class="grid-note">
        <span class="chip">Blue / White UI</span>
        <h3>重做信息结构</h3>
        <p>每个页面都拆成“说明区 + 配置区 + 监控区”，让操作顺序更直观。</p>
      </div>
      <div class="grid-note">
        <span class="chip">Focused Workflow</span>
        <h3>保留功能，替换表达</h3>
        <p>底层交互逻辑继续复用现有运行入口，前端布局、文案和视觉不再沿用旧实现。</p>
      </div>
    </aside>
  </section>
"""


FOOTER_HTML = """
  <section class="footer-shell">
    <p>
      <strong>SAGEM</strong><br>
      Multi-table entity matching experiment console.
    </p>
    <p>
      <span class="mono">Tip</span><br>
      先在对应工作区完成配置，再执行运行、生成或对比操作；结果页会自动汇总 <code>results/</code>。
    </p>
  </section>
</div>
"""


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    font=("IBM Plex Sans", "IBM Plex Sans SC", "sans-serif"),
    font_mono=("JetBrains Mono", "Consolas", "monospace"),
).set(
    body_background_fill="transparent",
    body_background_fill_dark="transparent",
    block_background_fill="transparent",
    block_border_width="0px",
    block_radius="0px",
    input_background_fill="#ffffff",
    input_border_color="#c9d9ea",
    button_primary_background_fill="#1e64c8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_border_color="#c9d9ea",
    button_secondary_text_color="#1e64c8",
    shadow_drop="none",
    shadow_drop_lg="none",
)


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="SAGEM Workbench",
        analytics_enabled=False,
        fill_width=True,
        theme=THEME,
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(HERO_HTML)

        with gr.Tabs(elem_classes="tab-nav"):
            with gr.Tab("主流程", id="tab-main"):
                tab_main_flow.create_tab()
            with gr.Tab("对比学习", id="tab-cl"):
                tab_contrastive.create_tab()
            with gr.Tab("数据生成", id="tab-llm"):
                tab_llm_gen.create_tab()
            with gr.Tab("结果分析", id="tab-results"):
                tab_results.create_tab()

        gr.HTML(FOOTER_HTML)

    return app


if __name__ == "__main__":
    ensure_supported_gradio()
    ensure_localhost_no_proxy()
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
