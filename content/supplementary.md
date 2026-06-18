# 補足実験: 欠損処理の選択と係数指標での比較

レポート本体（E3）は論文 [[1]](#ref1) の実データ設定に準拠し、欠損センチネル `-200` を残したまま予測誤差 RelMSE で
評価した。一方、実務で直観的に取りたくなる選択肢として「**欠損を含む行の削除**」「**欠損の補完**」、および
「**係数レベルの指標での比較**」がある。これらは妥当なアプローチなので、本体とは別に補足実験として実施し、
結果と注意点をここにまとめる（論文の主設定ではない）。

## 設定

UCI Air Quality（目的 CO(GT)）に対し、`-200`（欠損センチネル）の前処理と標準化を 4 通り用意して比較する。
bounds はいずれもデータ由来の non-private な min/max、予算は μ-GDP $\mu\in\{0.5,1,2,5\}$、20 反復平均。
スクリプト: `scripts/04_preprocessing_variants.py`。

| 変種 | 内容 | $n$ | $d$ | $\mathrm{cond}(X^\top X)$ | OLS RelMSE |
|---|---|---:|---:|---:|---:|
| keep（本体・論文準拠） | `-200` 残す（全 12 特徴） | 9357 | 12 | $7.4\times10^6$ | 0.441 |
| keep + standardized | keep を特徴標準化（z-score） | 9357 | 12 | $1.8\times10^4$ | 0.603 |
| drop_rows | NMHC(GT)（約 9 割欠損）を列除外＋`-200` 行を削除 | 6941 | 11 | $4.3\times10^8$ | 0.028 |
| impute | 特徴の `-200` を列平均で補完、目的が `-200` の行のみ削除 | 7674 | 12 | $4.4\times10^8$ | 0.035 |

## 指標

- **RelMSE**（予測誤差）$=\lVert X\hat\beta-y\rVert_2^2/\lVert y\rVert_2^2$。本体と同じ頑健な指標。
- **係数 relL2**（係数レベル）$=\lVert\hat\beta_{\text{DP}}-\hat\beta_{\text{OLS}}\rVert_2/\lVert\hat\beta_{\text{OLS}}\rVert_2$。
  直観的だが、後述のとおり悪条件な設計では不安定になる。

## 結果

![E4: RelMSE と係数 relL2 を前処理変種ごとに比較](results/figures/e4_preprocessing.png)

**BinAgg の予測 RelMSE（mean、μ別）:**

| 変種 | μ=0.5 | μ=1.0 | μ=2.0 | μ=5.0 |
|---|---:|---:|---:|---:|
| keep（本体） | 0.506 | 0.450 | 0.443 | 0.444 |
| keep + standardized | 0.617 | 0.608 | 0.608 | 0.607 |
| drop_rows | 0.076 | 0.050 | 0.036 | 0.032 |
| impute | 0.082 | 0.057 | 0.044 | 0.039 |

**DP–OLS 係数 relL2（mean±std、μ別）:**

| 変種 | μ=0.5 | μ=1.0 | μ=2.0 | μ=5.0 |
|---|---:|---:|---:|---:|
| keep（本体） | 0.97±0.16 | 0.91±0.10 | 0.96±0.13 | 0.84±0.16 |
| keep + standardized | 0.73±0.19 | 0.62±0.14 | 0.69±0.22 | 0.80±0.32 |
| drop_rows | 4.27±3.43 | 3.29±2.21 | 2.32±1.38 | 1.95±0.88 |
| impute | 3.42±3.56 | 2.07±1.74 | 2.19±1.19 | 1.50±0.83 |

## 観察と注意点

- **欠損を除く/補完すると予測 RelMSE は大きく下がる**（keep の ~0.45 に対し drop_rows/impute は ~0.03–0.08）。
  これは `-200` という極端な外れ値が消えて予測課題自体が易しくなるためで、手法が「良くなった」わけではない。
  絶対値が変わるので、**論文 Table 2（D7, keep 設定）と数値を比べるなら keep 設定を使う必要がある**。
- **係数 relL2 は概して大きく、ばらつきも大きい**。この実データの設計はどの前処理でも悪条件
  （$\mathrm{cond}(X^\top X)$ は keep で $7.4\times10^6$、欠損除去後はむしろ $4\times10^8$ 台。`-200` の外れ値が広がりを与えるため
  keep の方が条件数は小さい）で、係数が一部の悪条件方向に支配されるためである。
- **標準化（z-score）は条件数を約 400 分の 1（$7.4\times10^6\to1.8\times10^4$）に下げ、係数 relL2 を安定化させる**
  （keep の 0.9 前後 → 0.6〜0.8、ばらつきも縮小）。ただし本モデルは切片を持たないため、特徴を中心化すると
  予測の手がかりが減り、予測 RelMSE はかえって悪化する（0.44→0.60）。**条件数の改善と予測精度はトレードオフ**になりうる。
- まとめると、**予測誤差 RelMSE は前処理や悪条件に対して頑健**だが、**係数レベルの一致は脆い**。論文が実データで
  RelMSE を採るのは、DP 推定と OLS が同じ $y$ を予測することで比較の頑健性が得られるからだと解釈できる
  （用語と背景は[前処理とドメイン特化メモ](data-notes.html)）。係数レベルの評価が必要なら、標準化や次元削減で
  条件数を下げたうえで、切片を含むモデルにするなどの配慮が要る。

> 💡 実務メモ: 欠損処理の選択（残す／行削除／補完）は**予測誤差の絶対値**を大きく動かす。
> 手法間の比較や論文との照合では、前処理を固定して同条件で比べること。係数レベルの評価を行うなら、
> 標準化・共線性の整理・次元削減で設計の条件数を下げてから測るのが望ましい。

## 参照

<ol class="refs">
<li id="ref1">Lin, S., Slavković, A., &amp; Bhoomireddy, D. R. (2026). <em>Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees.</em> AISTATS 2026 (PMLR). arXiv:2510.16974. <a href="https://arxiv.org/abs/2510.16974">https://arxiv.org/abs/2510.16974</a></li>
<li id="ref2">BinAgg (Python package). Shuronglin/BinAgg, commit 13c09bb (2026-05-27). <a href="https://github.com/Shuronglin/BinAgg">https://github.com/Shuronglin/BinAgg</a></li>
<li id="ref3">UCI Machine Learning Repository — Air Quality Data Set（同梱 <code>data/AirQualityUCI.csv</code>）.</li>
</ol>
