# tpdp2026 Issue #16 — BinAgg 再現追試

論文 *Differentially Private Linear Regression and Synthetic Data Generation with
Statistical Guarantees* (Lin, Slavković & Bhoomireddy, AISTATS 2026,
[arXiv:2510.16974](https://arxiv.org/abs/2510.16974)) の DP 線形回帰＋統計的保証付き
合成データ生成を、公式実装 [BinAgg](https://github.com/Shuronglin/BinAgg) で再現追試する。

- **レポート本体**: [content/REPORT.md](content/REPORT.md)
- **実験計画**: [docs/plans/experiment-plan.md](docs/plans/experiment-plan.md)
- **元 Issue**: https://github.com/pwscup/tpdp2026/issues/16

## セットアップと実行

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt \
  "git+https://github.com/Shuronglin/BinAgg.git@13c09bb"
.venv/Scripts/python.exe scripts/01_sim_coverage.py      # E1: CI 被覆率（統計的保証）
.venv/Scripts/python.exe scripts/02_synthetic_utility.py # E2: 合成データの回帰 utility
.venv/Scripts/python.exe scripts/03_real_airquality.py   # E3: 実データ (Air Quality)
```

結果は `results/*.json|*.csv` と `results/figures/*.png` に出力される（シード固定で再現可能）。

## 構成

```
.
├── content/REPORT.md          … 追試レポート（Abstract〜結論・限界・References）
├── docs/plans/                … 実験計画（指標・ベースライン・予算）
├── scripts/                   … common.py + E1/E2/E3 実行スクリプト
├── data/AirQualityUCI.csv     … 実データ（BinAgg 同梱）
├── results/                   … JSON/CSV/図（実験出力）
├── requirements.txt           … 固定環境
└── .claude/                   … tabular-sdg-skillset（プロセススキル一式）
```

## 主な結果（詳細は REPORT.md）

| 実験 | 要点 |
|---|---|
| E1 | BinAgg の 95% CI 被覆率 0.92–0.93（公称 0.95 近接）／素朴 CI は 0.08–0.32 に崩壊 |
| E2 | 合成データへ素朴 OLS=相対誤差 0.58→0.06、BinAgg 補正=0.09→0.02（OLS(元) 0.003 に接近） |
| E3 | 実データで CI 幅・合成モーメント誤差が μ とともに単調縮小。`-200` 欠損センチネル汚染を特定 |
