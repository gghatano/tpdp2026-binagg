"""Build self-contained HTML pages from content/*.md into htmls/ for GitHub Pages.

Single source of truth for the site is the PAGES table below. Each Markdown page
is converted with tables/fenced-code/TOC; local images (results/figures/*.png) are
base64-embedded so each HTML file is shareable standalone. A sidebar table of
contents (from h2/h3) and a top nav across PAGES are injected.

Run:  .venv/Scripts/python.exe scripts/03_build_html.py
Output: htmls/<html>, plus htmls/.nojekyll
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
HTMLS = ROOT / "htmls"

# --- PAGES: single source of truth for the published site -------------------
PAGES = [
    {
        "md": "REPORT.md",
        "html": "index.html",
        "title": "BinAgg 再現追試レポート",
        "subtitle": "DP 線形回帰と統計的保証付き合成データ生成 (tpdp2026 Issue #16)",
    },
]

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml"}

CSS = """
:root{--fg:#1b1f24;--muted:#57606a;--bg:#fff;--line:#d0d7de;--accent:#0969da;
--codebg:#f6f8fa;--th:#f6f8fa}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,"Segoe UI",Helvetica,Arial,"Hiragino Kaku Gothic ProN","Noto Sans JP",sans-serif;
color:var(--fg);background:var(--bg);line-height:1.7}
.layout{display:flex;max-width:1180px;margin:0 auto;gap:32px;padding:0 20px}
nav.toc{position:sticky;top:0;align-self:flex-start;width:270px;max-height:100vh;
overflow:auto;padding:28px 0;font-size:13.5px}
nav.toc .brand{font-weight:700;font-size:15px;margin-bottom:4px}
nav.toc .sub{color:var(--muted);font-size:12px;margin-bottom:16px}
nav.toc ul{list-style:none;padding-left:0;margin:0}
nav.toc li{margin:2px 0}
nav.toc a{color:var(--muted);text-decoration:none;display:block;padding:3px 8px;border-left:2px solid transparent;border-radius:0 4px 4px 0}
nav.toc a:hover{color:var(--accent);background:var(--codebg)}
nav.toc ul ul a{padding-left:20px;font-size:12.5px}
main{flex:1;min-width:0;padding:28px 0 80px;max-width:820px}
main h1{font-size:28px;border-bottom:1px solid var(--line);padding-bottom:.3em;margin-top:0}
main h2{font-size:22px;border-bottom:1px solid var(--line);padding-bottom:.3em;margin-top:1.8em}
main h3{font-size:17px;margin-top:1.6em}
a{color:var(--accent)}
code{background:var(--codebg);padding:.15em .4em;border-radius:6px;font-size:85%;
font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
pre{background:var(--codebg);padding:14px;border-radius:8px;overflow:auto}
pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1em 0;display:block;overflow:auto}
th,td{border:1px solid var(--line);padding:6px 13px}
th{background:var(--th)}
tr:nth-child(2n){background:#f6f8fa55}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:8px 0}
blockquote{margin:1em 0;padding:0 1em;color:var(--muted);border-left:4px solid var(--line)}
hr{border:none;border-top:1px solid var(--line);margin:2em 0}
.footer{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}
@media(max-width:880px){nav.toc{display:none}.layout{padding:0 16px}}
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
<title>{title}</title>
{mathjax}
<style>{css}</style></head>
<body><div class="layout">
<nav class="toc"><div class="brand">{title}</div><div class="sub">{subtitle}</div>
{nav}{toc}</nav>
<main>{body}
<div class="footer">{subtitle} · build: scripts/03_build_html.py · source: <code>content/{md}</code></div>
</main></div></body></html>
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


def build_nav(current: str) -> str:
    if len(PAGES) <= 1:
        return ""
    items = "".join(
        f'<li><a href="{p["html"]}"{" style=color:var(--accent)" if p["html"]==current else ""}>{p["title"]}</a></li>'
        for p in PAGES
    )
    return f'<ul style="margin-bottom:16px">{items}</ul>'


def main() -> None:
    HTMLS.mkdir(parents=True, exist_ok=True)
    (HTMLS / ".nojekyll").write_text("", encoding="utf-8")

    for p in PAGES:
        src = CONTENT / p["md"]
        text = src.read_text(encoding="utf-8")
        md = markdown.Markdown(
            extensions=["tables", "fenced_code", "toc", "sane_lists",
                        "pymdownx.arithmatex"],
            extension_configs={"toc": {"toc_depth": "2-3"},
                               "pymdownx.arithmatex": {"generic": True}},
        )
        body = md.convert(text)
        body = embed_images(body)
        html = TEMPLATE.format(
            title=p["title"], subtitle=p["subtitle"], css=CSS, mathjax=MATHJAX,
            nav=build_nav(p["html"]), toc=md.toc, body=body, md=p["md"],
        )
        out = HTMLS / p["html"]
        out.write_text(html, encoding="utf-8")
        print(f"built {p['md']} -> htmls/{p['html']} ({len(html)//1024} KB)")

    print("wrote htmls/.nojekyll")


if __name__ == "__main__":
    main()
