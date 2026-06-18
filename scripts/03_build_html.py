"""Build the multi-page GitHub Pages site from content/*.md into htmls/.

Layout (modeled on gghatano.github.io/dpsynth-demo): a clean academic report on
the main tab (index) plus supplementary tabs (用語/手法/前処理/エンジニアリング/QA).
Each output is a self-contained HTML file: figures base64-embedded, math via
MathJax + pymdownx.arithmatex, a hero band + top tab nav (active tab highlighted)
and a sticky TOC sidebar built from h2/h3.

The PAGES table below is the single source of truth for the site (tabs/order).

Run:  .venv/Scripts/python.exe scripts/03_build_html.py
Output: htmls/<html> for each page, plus htmls/.nojekyll
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
HTMLS = ROOT / "htmls"

SITE_TITLE = "BinAgg 再現追試"

# --- PAGES: single source of truth for the site (tabs, order, metadata) -----
PAGES = [
    {"md": "index.md", "html": "index.html", "tab": "📄 レポート",
     "subtitle": "DP 線形回帰と統計的保証付き合成データ生成 — 再現追試 (tpdp2026 #16)"},
    {"md": "notation.md", "html": "notation.html", "tab": "📐 用語と記法",
     "subtitle": "本文で使う用語・記号の定義"},
    {"md": "method-binagg.md", "html": "method-binagg.html", "tab": "🧩 手法詳細",
     "subtitle": "BinAgg のアルゴリズムと直感"},
    {"md": "data-notes.md", "html": "data-notes.html", "tab": "🔬 前処理・データ",
     "subtitle": "前処理の工夫とドメイン知識"},
    {"md": "supplementary.md", "html": "supplementary.html", "tab": "🧪 補足実験",
     "subtitle": "欠損処理・標準化と係数指標での比較"},
    {"md": "extensions.md", "html": "extensions.html", "tab": "📈 発展実験",
     "subtitle": "比較手法 AdaSSP・他データセット・反復数"},
    {"md": "engineering-notes.md", "html": "engineering-notes.html", "tab": "🛠 エンジニアリング",
     "subtitle": "環境構築・再現手順・落とし穴"},
    {"md": "faq.md", "html": "faq.html", "tab": "❓ QA・メモ",
     "subtitle": "よくある疑問と設計判断"},
]

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml"}

CSS = """
:root{--fg:#1a1a1a;--muted:#666;--accent:#c0392b;--link:#1565c0;--line:#e3e3e3;
--bg:#f3f4f6;--card:#fff;--code:#f6f8fa;--sidebar:#fbfbfc;--chip:#eef4fb;--chipline:#cfe0f3}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:-apple-system,"Segoe UI","Hiragino Sans","Yu Gothic UI","Meiryo",sans-serif;line-height:1.85}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
/* hero + tabs */
.hero{background:linear-gradient(135deg,#2c3e50 0%,#c0392b 130%);color:#fff}
.hero .inner{max-width:1180px;margin:0 auto;padding:26px 20px 0}
.hero h1{margin:0 0 4px;font-size:1.6rem}
.hero p{margin:0 0 16px;opacity:.92;font-size:.95rem}
nav.tabs{max-width:1180px;margin:0 auto;padding:0 20px;display:flex;gap:6px;flex-wrap:wrap}
nav.tabs a{color:#fff;background:rgba(255,255,255,.14);padding:8px 14px;border-radius:8px 8px 0 0;
font-size:.9rem;white-space:nowrap}
nav.tabs a:hover{background:rgba(255,255,255,.26);text-decoration:none}
nav.tabs a.active{background:var(--bg);color:var(--accent);font-weight:600}
/* layout */
.layout{max-width:1180px;margin:0 auto;padding:24px 20px 40px;display:grid;
grid-template-columns:250px 1fr;gap:32px}
nav.toc{position:sticky;top:18px;align-self:start;max-height:calc(100vh - 36px);overflow:auto;
background:var(--sidebar);border:1px solid var(--line);border-radius:10px;padding:14px 16px;font-size:13px}
nav.toc .toc-title{font-weight:700;margin-bottom:6px}
nav.toc ul{list-style:none;margin:0;padding-left:0}
nav.toc li{margin:2px 0}
nav.toc a{color:var(--muted);display:block;padding:2px 6px;border-left:2px solid transparent;border-radius:0 4px 4px 0}
nav.toc a:hover{color:var(--accent);background:#fff;text-decoration:none}
nav.toc ul ul a{padding-left:18px;font-size:12.5px}
/* article card */
article{background:var(--card);border:1px solid var(--line);border-radius:10px;
box-shadow:0 1px 10px rgba(0,0,0,.04);padding:12px 40px 56px;min-width:0}
article>h1:first-of-type{display:none}
article :is(h2,h3,h4){scroll-margin-top:16px}
article h2{font-size:1.45rem;border-bottom:1px solid var(--line);padding-bottom:.3em;margin-top:1.9em}
article h3{font-size:1.16rem;margin-top:1.6em}
article h4{font-size:1rem;color:var(--accent)}
code{background:var(--code);padding:.12em .4em;border-radius:6px;font-size:88%;
font-family:"Cascadia Code",Consolas,"SF Mono",monospace}
pre{background:var(--code);padding:14px;border-radius:8px;overflow-x:auto}
pre code{background:none;padding:0}
table{border-collapse:collapse;width:100%;margin:1em 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 13px}
th{background:#f2f4f7}
tr:nth-child(2n){background:#fbfbfc}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:1.2em auto;display:block}
blockquote{margin:1em 0;padding:.6em 1em;border-left:4px solid var(--accent);
background:#fdf3f2;border-radius:0 8px 8px 0;color:#333}
hr{border:none;border-top:1px solid var(--line);margin:2.2em 0}
/* citation chips */
a[href^="#ref"]{font-size:.72em;vertical-align:super;line-height:0;background:var(--chip);
border:1px solid var(--chipline);border-radius:4px;padding:0 4px;color:var(--link);text-decoration:none}
a[href^="#ref"]:hover{background:#dceaf8}
ol.refs{font-size:.92rem;color:#333;padding-left:1.4em}
ol.refs li{margin:.4em 0}
ol.refs li:target{background:#fff6d6;border-radius:4px}
.footer{max-width:1180px;margin:0 auto;padding:8px 20px 40px;color:var(--muted);font-size:12px;text-align:center}
@media(max-width:860px){.layout{grid-template-columns:1fr}nav.toc{position:static;max-height:none}
article{padding:12px 18px 40px}}
"""

MATHJAX = """
<script>
window.MathJax = {
  tex: {inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]],
        processEscapes: true, processEnvironments: true},
  options: {ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex"}
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{ptitle} — {site}</title>
{mathjax}
<style>{css}</style></head>
<body>
<header class="hero"><div class="inner"><h1>{site}</h1><p>{subtitle}</p></div>
<nav class="tabs">{tabs}</nav></header>
<div class="layout">
<nav class="toc"><div class="toc-title">目次</div>{toc}</nav>
<article>{body}</article>
</div>
<div class="footer">{site} · {subtitle} · build: scripts/03_build_html.py · source: <code>content/{md}</code></div>
</body></html>
"""


def embed_images(html: str) -> str:
    def repl(m: re.Match) -> str:
        pre, src, post = m.group(1), m.group(2), m.group(3)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = (ROOT / src).resolve()
        if not path.exists():
            print(f"  WARN: image not found: {src}")
            return m.group(0)
        mime = MIME.get(path.suffix.lower(), "application/octet-stream")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img{pre}src="data:{mime};base64,{b64}"{post}>'

    return re.sub(r'<img([^>]*?)\ssrc="([^"]+)"([^>]*?)>', repl, html)


def build_tabs(current_html: str) -> str:
    out = []
    for p in PAGES:
        cls = ' class="active"' if p["html"] == current_html else ""
        out.append(f'<a href="{p["html"]}"{cls}>{p["tab"]}</a>')
    return "".join(out)


def page_title(p: dict) -> str:
    return p["tab"].split(" ", 1)[-1] if " " in p["tab"] else p["tab"]


def main() -> None:
    HTMLS.mkdir(parents=True, exist_ok=True)
    (HTMLS / ".nojekyll").write_text("", encoding="utf-8")

    for p in PAGES:
        src = CONTENT / p["md"]
        if not src.exists():
            print(f"  SKIP (missing): content/{p['md']}")
            continue
        text = src.read_text(encoding="utf-8")
        md = markdown.Markdown(
            extensions=["tables", "fenced_code", "toc", "sane_lists", "pymdownx.arithmatex"],
            extension_configs={"toc": {"toc_depth": "2-3"},
                               "pymdownx.arithmatex": {"generic": True}},
        )
        body = embed_images(md.convert(text))
        html = TEMPLATE.format(
            site=SITE_TITLE, ptitle=page_title(p), subtitle=p["subtitle"], css=CSS,
            mathjax=MATHJAX, tabs=build_tabs(p["html"]), toc=md.toc, body=body, md=p["md"],
        )
        (HTMLS / p["html"]).write_text(html, encoding="utf-8")
        print(f"built content/{p['md']} -> htmls/{p['html']} ({len(html)//1024} KB)")

    print("wrote htmls/.nojekyll")


if __name__ == "__main__":
    main()
