"""PathCL-EM 可视化界面 — Editorial/Magazine 风格"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
from web.tabs import tab_main_flow, tab_contrastive, tab_llm_gen, tab_results


# ---------------------------------------------------------------------------
# 全局样式:Editorial/Magazine — Ivory paper + 编辑红 + Fraunces 显示体
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..700,30..100;1,9..144,300..700,30..100&family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&family=JetBrains+Mono:wght@400;500;600&family=Noto+Serif+SC:wght@400;500;700&display=swap');

:root {
    /* Ivory paper */
    --paper:      #F5F1E8;
    --paper-sub:  #EDE8DC;
    --paper-deep: #E3DCCB;

    /* Ink */
    --ink:        #1A1916;
    --ink-soft:   #5A554C;
    --ink-faint:  #8B8477;

    /* Rules */
    --rule:       #CFC3AA;
    --rule-soft:  #E3DCCB;

    /* Accent: editorial vermillion */
    --accent:     #B8321A;
    --accent-deep:#8B1F0C;
    --accent-wash:#F4E0D9;
    --accent-ink: #F5F1E8;

    /* Secondary: forest */
    --secondary:  #2E4A38;

    /* Warn */
    --warn:       #8B4513;
    --warn-wash:  #F3E6D7;

    /* Ok (metrics/progress) */
    --ok:         #B8321A;

    /* Fonts */
    --serif:   "Fraunces", "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
    --display: "Fraunces", "Noto Serif SC", Georgia, serif;
    --sans:    "DM Sans", "Noto Sans SC", "PingFang SC", "Helvetica Neue", sans-serif;
    --mono:    "JetBrains Mono", "SF Mono", Consolas, monospace;
}

.dark {
    --paper:      #18140E;
    --paper-sub:  #221E16;
    --paper-deep: #2C271C;
    --ink:        #EDE8DC;
    --ink-soft:   #B8B0A0;
    --ink-faint:  #7A7469;
    --rule:       #3A3327;
    --rule-soft:  #2A251C;
    --accent:     #E06049;
    --accent-deep:#C1472F;
    --accent-wash:#2E1E18;
    --accent-ink: #18140E;
    --warn:       #C08550;
    --warn-wash:  #2A1F12;
}

/* ==== 纸面底色 + 极轻噪声颗粒 ==== */
html, body {
    background: var(--paper) !important;
    font-family: var(--sans);
    color: var(--ink);
}
body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 9999;
    opacity: 0.55;
    mix-blend-mode: multiply;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='260' height='260'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.92' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .042 0'/></filter><rect width='260' height='260' filter='url(%23n)'/></svg>");
}
.dark body::before { mix-blend-mode: screen; opacity: 0.25; }

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    background: transparent !important;
    font-family: var(--sans) !important;
    color: var(--ink) !important;
    padding: 0 !important;
}
/* Gradio 内部包裹层也放开宽度限制 */
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container .contain,
.gradio-container .app,
.gradio-container > div {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 32px !important;
    padding-right: 32px !important;
    box-sizing: border-box;
}
body, .gradio-container * { color: var(--ink); }

/* ==== 刊头(Editorial Cover) ==== */
.masthead {
    position: relative;
    padding: 36px 48px 22px 48px;
    margin: 0 -32px 8px -32px;
    border-bottom: 1px solid var(--ink);
    animation: fade-up 0.5s ease-out both;
}
.masthead::after {
    /* 双线底:一粗一细 */
    content: "";
    position: absolute;
    left: 0; right: 0;
    bottom: -5px;
    height: 2px;
    background: var(--accent);
}
.masthead .issue {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.72em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    padding: 5px 0;
    margin-bottom: 28px;
    border-top: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    font-weight: 600;
}
.masthead h1 {
    font-family: var(--display) !important;
    font-weight: 500;
    font-variation-settings: "opsz" 144, "SOFT" 40;
    font-size: 4.8em;
    line-height: 0.92;
    letter-spacing: -2.8px;
    margin: 0;
    color: var(--ink) !important;
}
.masthead h1 .mark {
    color: var(--accent) !important;
    font-style: italic;
    font-weight: 400;
    font-variation-settings: "opsz" 144, "SOFT" 100;
    margin: 0 2px;
}
.masthead .subtitle {
    font-family: var(--serif);
    font-style: italic;
    font-weight: 300;
    font-variation-settings: "opsz" 36;
    font-size: 1.32em;
    line-height: 1.4;
    color: var(--ink-soft);
    margin: 22px 0 6px 0;
    max-width: 36em;
}
.masthead .subtitle em {
    color: var(--accent);
    font-style: italic;
    font-weight: 400;
}
.masthead .byline {
    display: flex;
    gap: 26px;
    font-family: var(--mono);
    font-size: 0.7em;
    letter-spacing: 2.2px;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-top: 26px;
    padding-top: 14px;
    border-top: 1px solid var(--rule-soft);
    font-weight: 500;
}
.masthead .byline span::before { content: "·  "; color: var(--rule); }
.masthead .byline span:first-child::before { content: ""; }

/* ==== Tab 导航(刊物 Chapter Nav) ==== */
.tab-nav {
    border-bottom: 1px solid var(--rule) !important;
    padding: 0 !important;
    margin: 22px 0 32px 0 !important;
    gap: 0 !important;
    background: transparent !important;
    animation: fade-up 0.5s 0.08s ease-out both;
}
.tab-nav button {
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.78em !important;
    letter-spacing: 2.4px !important;
    text-transform: uppercase !important;
    padding: 12px 30px 14px 30px !important;
    border-radius: 0 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: var(--ink-faint) !important;
    transition: color 0.2s, border-color 0.2s !important;
    box-shadow: none !important;
    position: relative;
}
.tab-nav button::before {
    content: attr(data-ch);
    font-family: var(--display);
    font-size: 0.82em;
    font-style: italic;
    color: var(--accent);
    margin-right: 10px;
    letter-spacing: 0;
    text-transform: none;
    font-weight: 400;
}
.tab-nav button:hover { color: var(--ink) !important; }
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ==== 区块小标题(Numbered Section) ==== */
.section-h {
    font-family: var(--display);
    font-size: 1.35em;
    font-weight: 500;
    font-variation-settings: "opsz" 60;
    color: var(--ink);
    margin: 28px 0 6px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--rule);
    display: flex;
    align-items: baseline;
    gap: 14px;
    letter-spacing: -0.4px;
}
.section-h .idx {
    font-family: var(--mono);
    font-size: 0.6em;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    font-style: normal;
}
.section-desc {
    font-family: var(--serif);
    font-size: 0.97em;
    font-weight: 400;
    color: var(--ink-soft);
    margin: 4px 0 18px 0;
    line-height: 1.7;
    max-width: 46em;
}
.section-desc.dropcap::first-letter {
    font-family: var(--display);
    font-size: 3em;
    line-height: 0.85;
    float: left;
    padding: 8px 10px 0 0;
    color: var(--accent);
    font-weight: 500;
    font-variation-settings: "opsz" 144, "SOFT" 0;
}
.section-desc code {
    font-family: var(--mono);
    font-size: 0.92em;
    background: var(--paper-sub);
    padding: 1px 6px;
    border: 1px solid var(--rule-soft);
    color: var(--accent);
}

/* ==== Eyebrow 小标签(可复用) ==== */
.eyebrow {
    font-family: var(--mono);
    font-size: 0.7em;
    font-weight: 600;
    letter-spacing: 2.4px;
    text-transform: uppercase;
    color: var(--accent);
    display: inline-block;
    margin-bottom: 4px;
}

/* ==== 按钮:编辑红实心 + 偏移阴影 ==== */
button.primary, button[variant="primary"], .gr-button-primary {
    background: var(--accent) !important;
    color: var(--accent-ink) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.76em !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    box-shadow: 3px 3px 0 var(--ink) !important;
    transition: transform 0.12s, box-shadow 0.12s, background 0.15s !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: var(--accent-deep) !important;
    transform: translate(-1px, -1px);
    box-shadow: 4px 4px 0 var(--ink) !important;
}
button.primary:active, .gr-button-primary:active {
    transform: translate(2px, 2px);
    box-shadow: 1px 1px 0 var(--ink) !important;
}

button.secondary, .gr-button-secondary {
    background: var(--paper) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.76em !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 10.5px 22px !important;
    box-shadow: 3px 3px 0 var(--accent) !important;
    transition: transform 0.12s, box-shadow 0.12s !important;
}
button.secondary:hover {
    background: var(--accent-wash) !important;
    transform: translate(-1px, -1px);
    box-shadow: 4px 4px 0 var(--accent) !important;
}

button.stop, .gr-button-stop {
    background: var(--paper) !important;
    color: var(--warn) !important;
    border: 1.5px solid var(--warn) !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.76em !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 10.5px 22px !important;
    box-shadow: 3px 3px 0 var(--warn) !important;
    transition: transform 0.12s, box-shadow 0.12s !important;
}
button.stop:hover {
    background: var(--warn-wash) !important;
    transform: translate(-1px, -1px);
    box-shadow: 4px 4px 0 var(--warn) !important;
}

/* ==== 预设小按钮行 ==== */
.preset-row { gap: 10px !important; margin: 8px 0 4px 0 !important; }
.preset-row button {
    font-family: var(--mono) !important;
    font-size: 0.72em !important;
    font-weight: 600 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    padding: 8px 16px !important;
    border-radius: 0 !important;
    background: var(--paper-sub) !important;
    border: 1px solid var(--rule) !important;
    color: var(--ink-soft) !important;
    box-shadow: none !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
}
.preset-row button:hover {
    background: var(--accent-wash) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ==== 输入控件:下划线式 ==== */
input, textarea, select,
.gr-box, .gr-input, .gr-dropdown,
.gradio-textbox textarea,
.gradio-textbox input,
.gradio-dropdown {
    border-radius: 0 !important;
    box-shadow: none !important;
    font-family: var(--mono) !important;
    font-size: 0.9em !important;
}
.gr-form, .gr-box, .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.form {
    background: transparent !important;
    border: none !important;
}
label, .gradio-container label {
    font-family: var(--mono) !important;
    font-size: 0.72em !important;
    font-weight: 600 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
    margin-bottom: 4px !important;
}
label span, .gradio-container label span {
    color: var(--ink-soft) !important;
    font-family: var(--mono) !important;
}
/* info 小字 */
.gradio-container label + * small,
.gradio-container .info {
    font-family: var(--serif) !important;
    font-size: 0.85em !important;
    font-style: italic !important;
    font-weight: 300 !important;
    color: var(--ink-faint) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

/* ==== Accordion ==== */
.gr-accordion, details {
    border-radius: 0 !important;
    border: none !important;
    border-top: 1px solid var(--rule) !important;
    border-bottom: 1px solid var(--rule) !important;
    background: transparent !important;
    box-shadow: none !important;
    margin-bottom: 14px !important;
}
.gr-accordion > .label-wrap, summary {
    padding: 10px 2px !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.75em !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    background: transparent !important;
    border: none !important;
}

/* ==== 指标块(Editorial Stat Block) ==== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-top: 2px solid var(--ink);
    border-bottom: 2px solid var(--ink);
    margin: 12px 0 22px 0;
    background: var(--paper);
    position: relative;
}
.metric-grid::before {
    content: "METRICS";
    position: absolute;
    top: -9px;
    left: 12px;
    background: var(--paper);
    padding: 0 8px;
    font-family: var(--mono);
    font-size: 0.62em;
    letter-spacing: 2.5px;
    font-weight: 600;
    color: var(--accent);
}
.metric-cell {
    padding: 22px 22px 18px;
    border-right: 1px solid var(--rule);
    background: transparent;
    position: relative;
}
.metric-cell:last-child { border-right: none; }
.metric-cell .k {
    font-family: var(--mono);
    font-size: 0.66em;
    color: var(--accent);
    letter-spacing: 2.6px;
    margin-bottom: 10px;
    text-transform: uppercase;
    font-weight: 600;
}
.metric-cell .v {
    font-family: var(--display);
    font-size: 2.7em;
    font-weight: 400;
    font-variation-settings: "opsz" 144, "SOFT" 0;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -1.5px;
}
.metric-cell .v.dim {
    color: var(--ink-faint);
    font-weight: 300;
    font-style: italic;
    font-variation-settings: "opsz" 144, "SOFT" 80;
}
.metric-cell .note {
    font-family: var(--mono);
    font-size: 0.66em;
    color: var(--ink-faint);
    margin-top: 8px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ==== Datasheet ==== */
.info-sheet {
    border: none;
    border-left: 3px solid var(--accent);
    background: var(--paper-sub);
    padding: 14px 20px;
    margin: 4px 0 14px 0;
    font-family: var(--mono);
    font-size: 0.8em;
    line-height: 2;
    color: var(--ink-soft);
}
.info-sheet > div { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.info-sheet .label {
    display: inline-block;
    min-width: 82px;
    color: var(--ink-faint);
    font-size: 0.88em;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    font-weight: 600;
}
.info-sheet .value {
    color: var(--ink);
    font-family: var(--serif);
    font-size: 1.1em;
    font-weight: 500;
    letter-spacing: 0;
    text-transform: none;
}

/* ==== 状态行 ==== */
.status-line {
    padding: 9px 16px;
    font-family: var(--mono);
    font-size: 0.78em;
    letter-spacing: 1.5px;
    color: var(--ink-soft);
    border-left: 3px solid var(--rule);
    background: var(--paper-sub);
    margin: 6px 0 12px 0;
    text-transform: uppercase;
    font-weight: 500;
}
.status-line::before {
    content: "STATUS — ";
    color: var(--accent);
    margin-right: 6px;
    font-weight: 700;
    letter-spacing: 2px;
}
.status-line.on {
    border-left-color: var(--accent);
    color: var(--ink);
    background: var(--accent-wash);
}
.status-line.on::before { content: "LIVE — "; }
.status-line.on::after {
    content: " ●";
    color: var(--accent);
    margin-left: 8px;
    animation: pulse-dot 1.4s ease-in-out infinite;
}
.status-line.warn {
    border-left-color: var(--warn);
    color: var(--warn);
    background: var(--warn-wash);
}
.status-line.warn::before { content: "HALT — "; color: var(--warn); }

.progress-track {
    height: 2px;
    background: var(--rule-soft);
    margin-top: 10px;
    position: relative;
}
.progress-track::before,
.progress-track::after {
    content: "";
    position: absolute;
    top: -3px;
    width: 1px;
    height: 6px;
    background: var(--rule);
}
.progress-track::before { left: 50%; }
.progress-track::after  { left: 75%; }
.progress-bar {
    height: 100%;
    background: var(--accent);
    transition: width 0.3s cubic-bezier(.3, 1, .7, 1);
}

/* ==== 日志框 ==== */
.log-box textarea {
    font-family: var(--mono) !important;
    font-size: 0.8em !important;
    line-height: 1.75 !important;
    background: var(--paper-sub) !important;
    color: var(--ink) !important;
    border: none !important;
    border-left: 3px solid var(--rule) !important;
    border-top: 1px solid var(--rule) !important;
    border-bottom: 1px solid var(--rule) !important;
    border-right: 1px solid var(--rule-soft) !important;
    border-radius: 0 !important;
    padding: 14px 16px !important;
    scrollbar-width: thin;
}
.dark .log-box textarea {
    background: var(--paper-sub) !important;
}

/* ==== 表格 ==== */
.gradio-dataframe, .gr-dataframe {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.gradio-dataframe table, .gr-dataframe table {
    border-radius: 0 !important;
    font-family: var(--serif) !important;
    font-size: 0.92em !important;
    border-collapse: collapse !important;
    border-top: 2px solid var(--ink) !important;
    border-bottom: 2px solid var(--ink) !important;
    background: transparent !important;
}
.gradio-dataframe th, .gr-dataframe th {
    background: var(--paper-sub) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.72em !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-bottom: 1.5px solid var(--accent) !important;
    padding: 10px 12px !important;
    text-align: left !important;
}
.gradio-dataframe td, .gr-dataframe td {
    border-bottom: 1px solid var(--rule-soft) !important;
    padding: 9px 12px !important;
    color: var(--ink) !important;
    font-family: var(--serif) !important;
    font-variation-settings: "opsz" 12;
    font-size: 1em !important;
    background: transparent !important;
}
.gradio-dataframe tr:hover td {
    background: var(--accent-wash) !important;
    cursor: pointer;
}

/* ==== 代码块(结果详情 JSON) ==== */
.gr-code, .code_wrap, .language-json {
    background: var(--paper-sub) !important;
    border: 1px solid var(--rule) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 0 !important;
}
.gr-code pre, .gr-code code {
    font-family: var(--mono) !important;
    font-size: 0.8em !important;
    background: transparent !important;
}

/* ==== Plot ==== */
.gradio-plot, .gr-plot {
    background: var(--paper-sub) !important;
    border: 1px solid var(--rule) !important;
    border-top: 2px solid var(--ink) !important;
    border-radius: 0 !important;
    padding: 10px !important;
}

/* ==== 页脚 ==== */
.colophon {
    margin-top: 64px;
    padding: 24px 0 34px 0;
    border-top: 1px solid var(--ink);
    text-align: center;
    position: relative;
}
.colophon::before {
    content: "";
    position: absolute;
    left: 0; right: 0;
    top: 4px;
    height: 1px;
    background: var(--accent);
}
.colophon .mark-fin {
    font-family: var(--display);
    font-style: italic;
    font-variation-settings: "opsz" 72;
    font-size: 1.4em;
    color: var(--ink);
    letter-spacing: 4px;
    margin-bottom: 8px;
    font-weight: 300;
}
.colophon .mark-fin::before { content: "— "; color: var(--accent); }
.colophon .mark-fin::after  { content: " —"; color: var(--accent); }
.colophon .colophon-text {
    font-family: var(--serif);
    font-size: 0.82em;
    font-style: italic;
    color: var(--ink-faint);
    line-height: 1.9;
    letter-spacing: 0.3px;
}
.colophon .colophon-text em {
    font-style: italic;
    color: var(--accent);
    font-weight: 500;
}
.colophon .mono {
    font-family: var(--mono);
    font-style: normal;
    font-size: 0.92em;
    letter-spacing: 1.2px;
    color: var(--ink-soft);
}

/* ==== 动画 ==== */
@keyframes fade-up {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
}

/* 内容区载入动画(稍迟于 masthead) */
.gradio-container .tabitem {
    animation: fade-up 0.45s 0.12s ease-out both;
}

/* ==== 滚动条 ==== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--paper-sub); }
::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--ink-faint); }

/* ==== Slider/Number 外观统一 ==== */
.gradio-slider input[type="range"] {
    accent-color: var(--accent);
}
.gradio-number input, .gradio-textbox input {
    border: none !important;
    border-bottom: 1.5px solid var(--rule) !important;
    background: transparent !important;
    font-family: var(--mono) !important;
    padding: 6px 2px !important;
    transition: border-color 0.15s;
}
.gradio-number input:focus, .gradio-textbox input:focus {
    border-bottom-color: var(--accent) !important;
    outline: none !important;
}

/* ==== Radio/Checkbox ==== */
.gradio-radio label, .gradio-checkbox label {
    font-family: var(--serif) !important;
    font-size: 0.95em !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    color: var(--ink) !important;
}
.gradio-radio input[type="radio"]:checked + *,
.gradio-checkbox input[type="checkbox"]:checked + * {
    accent-color: var(--accent);
}

/* ==== Dropdown ==== */
.gradio-dropdown .wrap, .gradio-dropdown input {
    border: none !important;
    border-bottom: 1.5px solid var(--rule) !important;
    background: transparent !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 0.92em !important;
    letter-spacing: 1px;
}
.gradio-dropdown .wrap:focus-within,
.gradio-dropdown input:focus {
    border-bottom-color: var(--accent) !important;
}
"""


# ---------------------------------------------------------------------------
# 刊头 / 页脚 HTML
# ---------------------------------------------------------------------------
MASTHEAD_HTML = """
<div class="masthead">
  <div class="issue">
    <span>Vol. 03 &nbsp;·&nbsp; № 01</span>
    <span>Apr &nbsp;·&nbsp; 2026</span>
  </div>
  <h1>PathCL<span class="mark">·</span>EM</h1>
  <div class="subtitle">
    <em>Contrastive learning</em> and <em>LLM-driven augmentation</em><br>
    for multi-table entity matching at scale.
  </div>
  <div class="byline">
    <span>A visualization workbench</span>
    <span>Multi-Table EM</span>
    <span>Gradio 6.x</span>
    <span>Anonymous · Submission</span>
  </div>
</div>
"""

COLOPHON_HTML = """
<div class="colophon">
  <div class="mark-fin">fin</div>
  <div class="colophon-text">
    Printed on ivory &nbsp;·&nbsp; set in <em>Fraunces</em> &amp; <em>DM Sans</em><br>
    <span class="mono">Tip — 按数据集一键预设参数 · 点击结果表格行查看详情</span>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# 主题 — 基底保持中性,细节由自定义 CSS 覆盖
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=("DM Sans", "Noto Sans SC", "PingFang SC", "sans-serif"),
    font_mono=("JetBrains Mono", "Consolas", "monospace"),
).set(
    body_background_fill="var(--paper)",
    body_background_fill_dark="var(--paper)",
    block_background_fill="transparent",
    block_border_color="var(--rule)",
    block_border_width="0px",
    block_radius="0px",
    input_background_fill="transparent",
    input_border_color="var(--rule)",
    input_radius="0px",
    button_primary_background_fill="var(--accent)",
    button_primary_text_color="var(--accent-ink)",
    button_secondary_background_fill="var(--paper)",
    button_secondary_border_color="var(--accent)",
    button_secondary_text_color="var(--accent)",
    shadow_drop="none",
    shadow_drop_lg="none",
    shadow_spread="0px",
)


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="PathCL-EM · Editorial Workbench",
        analytics_enabled=False,
        fill_width=True,
    ) as app:
        gr.HTML(MASTHEAD_HTML)

        with gr.Tabs():
            with gr.Tab("主流程 · Process", id="tab-main"):
                tab_main_flow.create_tab()
            with gr.Tab("对比学习 · Contrastive", id="tab-cl"):
                tab_contrastive.create_tab()
            with gr.Tab("数据生成 · Synthesis", id="tab-llm"):
                tab_llm_gen.create_tab()
            with gr.Tab("结果 · Archive", id="tab-results"):
                tab_results.create_tab()

        gr.HTML(COLOPHON_HTML)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=THEME,
        css=CUSTOM_CSS,
    )
