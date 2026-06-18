# tpdp2026 Issue #16 — BinAgg 再現追試

論文 *Differentially Private Linear Regression and Synthetic Data Generation with
Statistical Guarantees* (Lin, Slavković & Bhoomireddy, AISTATS 2026,
[arXiv:2510.16974](https://arxiv.org/abs/2510.16974)) の DP 線形回帰＋統計的保証付き
合成データ生成を、公式実装 [BinAgg](https://github.com/Shuronglin/BinAgg) で再現追試する。

- **公開サイト (GitHub Pages)**: https://gghatano.github.io/tpdp2026-binagg/
- **元 Issue**: https://github.com/pwscup/tpdp2026/issues/16

公開サイトは「論文体のクリーンな本体（レポート）＋補助タブ」の二層構成。本文は本体に、用語・手法・前処理・
環境・QA などの補足は別タブに分けてある。

| ページ (タブ) | Markdown | 役割 |
|---|---|---|
| 📄 レポート | [content/index.md](content/index.md) | 論文本体（アブストラクト〜結論・参考文献） |
| 📐 用語と記法 | [content/notation.md](content/notation.md) | 用語・記号の定義 |
| 🧩 手法詳細 | [content/method-binagg.md](content/method-binagg.md) | BinAgg のアルゴリズムと直感 |
| 🔬 前処理・データ | [content/data-notes.md](content/data-notes.md) | 前処理の工夫・ドメイン知識・`-200` 知見 |
| 🧪 補足実験 | [content/supplementary.md](content/supplementary.md) | 欠損処理・標準化と係数指標での比較 |
| 📈 発展実験 | [content/extensions.md](content/extensions.md) | 比較手法 AdaSSP・他データ(Abalone/Wine/Appliances)・反復数 |
| 🛠 エンジニアリング | [content/engineering-notes.md](content/engineering-notes.md) | 環境構築・再現手順・落とし穴 |
| ❓ QA・メモ | [content/faq.md](content/faq.md) | よくある疑問と設計判断 |

- **実験計画**: [docs/plans/experiment-plan.md](docs/plans/experiment-plan.md)

## セットアップと実行

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt \
  "git+https://github.com/Shuronglin/BinAgg.git@13c09bb"
.venv/Scripts/python.exe scripts/01_sim_coverage.py      # E1: CI 被覆率（統計的保証）
.venv/Scripts/python.exe scripts/02_synthetic_utility.py # E2: 合成データの回帰 utility
.venv/Scripts/python.exe scripts/03_real_airquality.py   # E3: 実データ (論文準拠 D7, RelMSE)
.venv/Scripts/python.exe scripts/04_preprocessing_variants.py # 補足: 欠損処理・標準化・係数指標
.venv/Scripts/python.exe scripts/05_other_datasets.py    # 発展: 他データ(Abalone/Wine/Appliances)
.venv/Scripts/python.exe scripts/06_adassp.py            # 発展: AdaSSP 比較 (D7)
```

結果は `results/*.json|*.csv` と `results/figures/*.png` に出力される（シード固定で再現可能）。

## サイトのビルド & 公開

```bash
uv pip install --python .venv markdown pymdown-extensions   # ビルド依存
.venv/Scripts/python.exe scripts/03_build_html.py   # content/*.md -> htmls/*.html
```

ビルダーは各 Markdown を自己完結 HTML 化する（図は base64 埋め込み、数式は MathJax + arithmatex、
ヒーロー＋上部タブ＋目次サイドバーを注入）。`main` への push で `.github/workflows/deploy-pages.yml` が
`htmls/` を GitHub Pages へデプロイする。**ページ構成（タブ・順序）は `scripts/03_build_html.py` の
`PAGES` テーブルが単一の真実**。`htmls/` は成果物なので直接編集しない。

## 構成

```
.
├── content/                   … 手で書く Markdown（index=本体 + 補助5ページ）
├── htmls/                     … ビルド成果物（直接編集しない・Pages 公開元）
├── docs/plans/                … 実験計画（指標・ベースライン・予算）
├── scripts/                   … common.py + E1/E2/E3 + 03_build_html.py（サイトビルダー）
├── data/AirQualityUCI.csv     … 実データ（BinAgg 同梱）
├── results/                   … JSON/CSV/図（実験出力）
├── requirements.txt           … 固定環境
└── .claude/                   … tabular-sdg-skillset（プロセススキル一式）
```

## 主な結果（詳細は公開サイトの 📄 レポート）

| 実験 | 要点 |
|---|---|
| E1 | BinAgg の 95% CI 被覆率 0.92–0.93（公称 0.95 近接）／素朴 CI は 0.08–0.32 に崩壊 |
| E2 | 合成データへ素朴 OLS=相対誤差 0.58→0.06、BinAgg 補正=0.09→0.02（OLS(元) 0.003 に接近） |
| E3 | 論文準拠(D7=Air Quality, n=9357,d=12, 予測誤差RelMSE, non-private bounds)。μ=1 で OLS=0.441(論文と完全一致)・BinAgg=0.450(論文0.463)を**再現** |
