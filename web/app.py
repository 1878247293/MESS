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
    width: min(1440px, calc(100vw - 32px));
    margin: 12px auto 16px;
    padding: 16px 18px;
    border: 1px solid rgba(156, 183, 216, 0.55);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.84);
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
}

.hero {
    display: block;
    margin-bottom: 12px;
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
    box-shadow: 0 4px 14px rgba(26, 70, 121, 0.04);
}

.hero-panel {
    padding: 14px 18px;
    border-radius: 14px;
}

.hero-kicker,
.section-kicker,
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 999px;
    background: var(--primary-soft);
    color: var(--primary);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero-title {
    margin: 8px 0 6px;
    font-size: clamp(18px, 2.2vw, 26px);
    line-height: 1.15;
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
    font-size: 12px;
    line-height: 1.6;
    color: var(--text-soft);
}

.hero-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
}

.hero-metrics .hero-stat {
    padding: 8px 12px;
    border-radius: 10px;
    background: rgba(220, 234, 254, 0.48);
    border: 1px solid rgba(156, 183, 216, 0.42);
}

.hero-stat .num {
    display: block;
    font-size: 18px;
    line-height: 1;
    font-weight: 700;
    color: var(--primary);
}

.hero-stat .label {
    display: block;
    margin-top: 4px;
    font-size: 11px;
    color: var(--text-faint);
}

.hero-side {
    display: grid;
    gap: 10px;
    padding: 14px 16px;
    border-radius: 14px;
}

.hero-side h3,
.page-card h3 {
    margin: 0;
    font-size: 13px;
    line-height: 1.3;
}

.hero-side p,
.page-card p,
.section-desc {
    margin: 0;
    font-size: 12px;
    line-height: 1.55;
    color: var(--text-soft);
}

.grid-note {
    display: grid;
    gap: 6px;
}

.layout-row {
    gap: 0 !important;
    align-items: flex-start !important;
}

.side-nav {
    position: fixed !important;
    top: 96px;
    left: 16px;
    z-index: 999;
    user-select: none;
    cursor: move;
    min-width: 180px !important;
    max-width: 220px !important;
    background: rgba(235, 244, 253, 0.95);
    border: 1px solid rgba(156, 183, 216, 0.55);
    border-radius: 12px;
    padding: 8px !important;
    box-shadow: 0 10px 28px rgba(26, 70, 121, 0.12);
    transition: min-width 0.22s ease, max-width 0.22s ease, padding 0.22s ease, box-shadow 0.18s ease;
}

.side-nav::before {
    content: "";
    display: block;
    width: 28px;
    height: 3px;
    margin: 0 auto 6px;
    border-radius: 2px;
    background: rgba(156, 183, 216, 0.55);
}

.side-nav.dragging {
    box-shadow: 0 16px 40px rgba(26, 70, 121, 0.22);
    transition: none !important;
}

.side-nav .nav-item button,
.side-nav button.nav-item,
.side-nav .nav-item {
    width: 100% !important;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 8px 12px !important;
    margin: 3px 0 !important;
    border-radius: 8px !important;
    min-height: 34px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background 0.18s ease, color 0.18s ease;
}

.side-nav button.secondary {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-soft) !important;
    box-shadow: none !important;
}

.side-nav button.secondary:hover {
    background: rgba(255, 255, 255, 0.78) !important;
    color: var(--text) !important;
}

.side-nav button.primary {
    background: linear-gradient(180deg, rgba(255, 255, 255, 1), rgba(226, 239, 252, 0.96)) !important;
    color: var(--primary) !important;
    border: 1px solid rgba(156, 183, 216, 0.62) !important;
    box-shadow: none !important;
}

.sidebar-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 6px 12px;
    margin-bottom: 6px;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(156, 183, 216, 0.55);
    color: var(--text-soft);
    border-radius: 8px;
    cursor: pointer;
    font-family: var(--sans);
    font-size: 12px;
    font-weight: 600;
    transition: background 0.18s, color 0.18s;
}

.sidebar-toggle:hover {
    background: var(--primary-soft);
    color: var(--primary);
}

.sidebar-toggle .bar {
    font-size: 14px;
    line-height: 1;
}

.layout-row.sidebar-collapsed .side-nav {
    min-width: 48px !important;
    max-width: 48px !important;
    padding: 6px !important;
}

.layout-row.sidebar-collapsed .side-nav .nav-item,
.layout-row.sidebar-collapsed .side-nav button.nav-item {
    padding: 8px 6px !important;
    justify-content: center !important;
    overflow: hidden;
    white-space: nowrap;
    font-size: 0 !important;
}

.layout-row.sidebar-collapsed .side-nav .nav-item::first-letter,
.layout-row.sidebar-collapsed .side-nav button.nav-item::first-letter {
    font-size: 13px !important;
}

.layout-row.sidebar-collapsed .sidebar-toggle .text {
    display: none;
}

.layout-row.sidebar-collapsed .sidebar-toggle {
    justify-content: center;
    padding: 6px;
}

.main-pane {
    min-width: 0 !important;
}

/* 隐藏 Gradio 默认 footer (通过 API 使用 / 使用 Gradio 构建 / 设置) */
.gradio-container footer,
.gradio-container > .main > footer,
.gradio-container > footer {
    display: none !important;
}

/* legacy gr.Tabs styling kept in case something else uses it */
.tab-nav {
    gap: 4px !important;
    margin: 0 0 12px !important;
    padding: 4px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(156, 183, 216, 0.48) !important;
    background: rgba(235, 244, 253, 0.75) !important;
}

.tab-nav button {
    min-height: 34px !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: var(--text-soft) !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease, transform 0.18s ease !important;
}

.tab-nav button:hover {
    background: rgba(255, 255, 255, 0.78) !important;
    color: var(--text) !important;
}

.tab-nav button.selected {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(226, 239, 252, 0.96)) !important;
    color: var(--primary) !important;
    border-color: rgba(156, 183, 216, 0.65) !important;
}

.tab-nav button span {
    font-size: 13px !important;
    font-weight: 600 !important;
}

.workspace {
    padding: 2px 0 6px;
}

.page-intro {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(240px, 0.8fr);
    gap: 10px;
    margin-bottom: 12px;
}

.page-card {
    padding: 12px 14px;
    border-radius: 12px;
}

.page-card.compact {
    display: grid;
    align-content: start;
    gap: 6px;
}

.section-title {
    margin: 6px 0 4px;
    font-size: 18px;
    line-height: 1.2;
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
    gap: 8px;
    margin: 0 0 8px;
    font-size: 13px;
    line-height: 1.3;
    font-weight: 700;
    color: var(--text);
}

.section-h .idx {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 7px;
    background: var(--primary-soft);
    color: var(--primary);
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
}

.panel-muted {
    padding: 8px 12px;
    border-radius: 10px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 8px 0 12px;
}

.metric-cell {
    padding: 10px 14px;
    border-radius: 12px;
}

.metric-cell .k {
    display: block;
    font-size: 11px;
    color: var(--text-faint);
    font-family: var(--mono);
    text-transform: uppercase;
}

.metric-cell .v {
    display: block;
    margin-top: 4px;
    font-size: 22px;
    line-height: 1;
    font-weight: 700;
    color: var(--primary);
}

.metric-cell .v.dim {
    color: var(--text-faint);
}

.metric-cell .note {
    margin-top: 4px;
    font-size: 11px;
    color: var(--text-faint);
}

.info-sheet {
    display: grid;
    gap: 6px;
    padding: 10px 12px;
    border-radius: 12px;
    color: var(--text-soft);
    font-size: 12px;
}

.info-sheet > div {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px;
}

.info-sheet .label {
    min-width: 72px;
    color: var(--text-faint);
    font-size: 11px;
    font-family: var(--mono);
    text-transform: uppercase;
}

.info-sheet .value {
    color: var(--text);
    font-weight: 600;
}

.status-line {
    padding: 10px 12px;
    border-radius: 12px;
    color: var(--text);
    font-size: 12px;
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
    height: 6px;
    margin-top: 8px;
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
    font-size: 12px !important;
    line-height: 1.5 !important;
}

.gr-form, .gr-box, .block, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

label, .gradio-container label {
    font-family: var(--sans) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text-soft) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

.gradio-container .info,
.gradio-container label + * small {
    color: var(--text-faint) !important;
    font-size: 11px !important;
}

input:not([type="checkbox"]):not([type="radio"]), textarea, select,
.gr-box, .gr-input, .gr-dropdown,
.gradio-textbox textarea,
.gradio-textbox input,
.gradio-dropdown .wrap,
.gradio-number input {
    border-radius: 10px !important;
    border: 1px solid rgba(156, 183, 216, 0.62) !important;
    background: rgba(255, 255, 255, 0.92) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    font-size: 12px !important;
}

input:focus, textarea:focus,
.gradio-dropdown .wrap:focus-within,
.gradio-number input:focus {
    border-color: var(--primary) !important;
}

.gradio-slider input[type="range"] {
    accent-color: var(--primary);
}

input[type="checkbox"], input[type="radio"] {
    accent-color: var(--primary);
    width: 15px !important;
    height: 15px !important;
    cursor: pointer;
}

.gr-accordion, details {
    border: 1px solid rgba(156, 183, 216, 0.48) !important;
    border-radius: 12px !important;
    background: rgba(245, 250, 255, 0.92) !important;
    overflow: hidden !important;
}

.gr-accordion > .label-wrap, summary {
    padding: 8px 12px !important;
    color: var(--primary) !important;
    font-family: var(--sans) !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    background: transparent !important;
}

button.primary, button[variant="primary"], .gr-button-primary {
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    background: linear-gradient(180deg, #2b79e8 0%, var(--primary) 100%) !important;
    color: var(--primary-ink) !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    box-shadow: 0 6px 14px rgba(30, 100, 200, 0.18) !important;
    min-height: 34px !important;
}

button.secondary, .gr-button-secondary,
button.stop, .gr-button-stop,
.preset-row button {
    border-radius: 10px !important;
    font-family: var(--sans) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    box-shadow: none !important;
    min-height: 34px !important;
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
    gap: 8px !important;
    margin-top: 6px !important;
}

.footer-shell {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-top: 12px;
    padding: 10px 14px;
    border-radius: 12px;
}

.footer-shell p {
    margin: 0;
    color: var(--text-soft);
    font-size: 11px;
    line-height: 1.55;
}

.mono {
    font-family: var(--mono);
}

code {
    padding: 1px 5px;
    border-radius: 6px;
    background: rgba(220, 234, 254, 0.7);
    color: var(--primary-deep);
    font-family: var(--mono);
    font-size: 11px;
}

table {
    font-size: 12px !important;
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
        width: min(100vw - 12px, 100%);
        margin: 6px auto 12px;
        padding: 10px;
        border-radius: 14px;
    }

    .hero-panel,
    .hero-side,
    .page-card,
    .footer-shell {
        padding: 12px;
        border-radius: 12px;
    }

    .section-title {
        font-size: 16px;
    }

    .footer-shell {
        flex-direction: column;
    }
}
"""


HERO_HTML = """
<div class="app-shell">
  <section class="hero hero-slim">
    <div class="hero-panel">
      <span class="hero-kicker">SAGEM Workbench</span>
      <h1 class="hero-title">面向 <span class="accent">多表实体匹配</span> 的统一实验前端</h1>
      <p class="hero-copy">主流程、对比学习、LLM 数据生成、结果回看四块工作区集中在此，所有运行入口共用同一套配置和日志反馈。</p>
    </div>
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


DRAG_SIDEBAR_SCRIPT = """
<script>
(function() {
    function isInteractive(el) {
        if (!el) return false;
        if (el.tagName === 'BUTTON' || el.tagName === 'A' || el.tagName === 'INPUT' || el.tagName === 'LABEL') return true;
        return !!(el.closest && el.closest('button, a, input, label'));
    }

    function init(nav) {
        if (!nav || nav.dataset.dragInit) return;
        nav.dataset.dragInit = '1';

        var dragging = false, moved = false;
        var startX = 0, startY = 0, initLeft = 0, initTop = 0;

        nav.addEventListener('mousedown', function(e) {
            if (isInteractive(e.target)) return;
            dragging = true;
            moved = false;
            startX = e.clientX;
            startY = e.clientY;
            var rect = nav.getBoundingClientRect();
            initLeft = rect.left;
            initTop = rect.top;
            nav.style.left = initLeft + 'px';
            nav.style.top = initTop + 'px';
            nav.style.right = 'auto';
            nav.classList.add('dragging');
            e.preventDefault();
        });

        document.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var dx = e.clientX - startX;
            var dy = e.clientY - startY;
            if (!moved && (Math.abs(dx) > 2 || Math.abs(dy) > 2)) moved = true;
            var newLeft = initLeft + dx;
            var newTop = initTop + dy;
            var maxLeft = window.innerWidth - nav.offsetWidth - 8;
            var maxTop = window.innerHeight - nav.offsetHeight - 8;
            nav.style.left = Math.max(8, Math.min(maxLeft, newLeft)) + 'px';
            nav.style.top = Math.max(8, Math.min(maxTop, newTop)) + 'px';
        });

        document.addEventListener('mouseup', function() {
            if (dragging) {
                dragging = false;
                nav.classList.remove('dragging');
            }
        });
    }

    function attempt() {
        var nav = document.querySelector('.side-nav');
        if (nav) init(nav);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attempt);
    } else {
        attempt();
    }

    var mo = new MutationObserver(function() { attempt(); });
    mo.observe(document.body, { childList: true, subtree: true });
})();
</script>
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
        head=DRAG_SIDEBAR_SCRIPT,
    ) as app:
        gr.HTML(HERO_HTML)

        with gr.Row(elem_classes="layout-row"):
            with gr.Column(scale=0, min_width=180, elem_classes="side-nav"):
                gr.HTML(
                    '<button type="button" class="sidebar-toggle"'
                    ' onclick="document.querySelector(\'.layout-row\').classList.toggle(\'sidebar-collapsed\'); return false;">'
                    '<span class="bar">≡</span><span class="text">收起菜单</span>'
                    '</button>'
                )
                nav_main = gr.Button("主流程", variant="primary", elem_classes="nav-item")
                nav_cl = gr.Button("对比学习", variant="secondary", elem_classes="nav-item")
                nav_llm = gr.Button("数据生成", variant="secondary", elem_classes="nav-item")
                nav_results = gr.Button("结果分析", variant="secondary", elem_classes="nav-item")

            with gr.Column(scale=10, elem_classes="main-pane"):
                with gr.Column(visible=True, elem_classes="pane") as pane_main:
                    tab_main_flow.create_tab()
                with gr.Column(visible=False, elem_classes="pane") as pane_cl:
                    tab_contrastive.create_tab()
                with gr.Column(visible=False, elem_classes="pane") as pane_llm:
                    tab_llm_gen.create_tab()
                with gr.Column(visible=False, elem_classes="pane") as pane_results:
                    tab_results.create_tab()

        gr.HTML(FOOTER_HTML)

        nav_buttons = [nav_main, nav_cl, nav_llm, nav_results]
        panes = [pane_main, pane_cl, pane_llm, pane_results]

        def _make_selector(idx: int):
            def _fn():
                pane_updates = [gr.update(visible=(i == idx)) for i in range(4)]
                btn_updates = [
                    gr.update(variant="primary" if i == idx else "secondary")
                    for i in range(4)
                ]
                return pane_updates + btn_updates

            return _fn

        for i, btn in enumerate(nav_buttons):
            btn.click(fn=_make_selector(i), outputs=panes + nav_buttons)

    return app


if __name__ == "__main__":
    ensure_supported_gradio()
    ensure_localhost_no_proxy()
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
