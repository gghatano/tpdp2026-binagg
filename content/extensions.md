# 発展実験: 比較手法と他データセット

レポート本体（E3）は論文 [[1]](#ref1) の D7（Air Quality）を予測誤差 RelMSE で再現した。ここでは
カバレッジを広げる発展実験として、(1) 比較手法 **AdaSSP** の追加、(2) 同定できた**他データセット 3 件**での
追試、(3) 反復回数の影響、(4) **合成データの下流 ML 評価（論文 §5.2.2 / Table 3）**を best-effort でまとめる。
(1)–(3) は論文 Table 2、(4) は Table 3 の報告値と並べて評価する。

> 📘 best-effort の注記: 競合手法の再実装や他データの前処理は、論文 Appendix の詳細が完全には公開されていないため、
> 絶対値の厳密一致は保証しない。優劣の傾向と桁を確認する目的で実施する。

## 1. 比較手法: AdaSSP（D7, μ=1, 100 反復）

論文は BinAgg を AdaSSP（Wang 2018）・DP-GD と比較する。いずれも BinAgg パッケージ外のため、AdaSSP を
μ-GDP 版で自前実装し（`scripts/06_adassp.py`、3 つの Gaussian 機構に予算分割）、D7 で比較した。DP-GD は
チューニング感度が高く高コストと論文も述べているため対象外とする。

![E6: D7 method comparison (RelMSE)](results/figures/e6_adassp.png)

| 手法 | 本追試 RelMSE (μ=1) | 論文 D7 報告値 |
|---|---:|---:|
| 非プライベート OLS | 0.441 | 0.441 |
| BinAgg | 0.450 | 0.463 |
| AdaSSP（best-effort） | 0.595 | 0.682 |
| DP-GD | — (対象外) | 0.852 |

- **事実**: 手法の優劣は **BinAgg < AdaSSP**（RelMSE が小さいほど良い）で、論文の順序（BinAgg 0.463 < AdaSSP 0.682）を
  **質的に再現**した。絶対値（AdaSSP 0.595 vs 論文 0.682）は best-effort 実装のため差があるが、桁と順序は整合する。
- **考察**: BinAgg が AdaSSP より良い予測精度を与えるという論文の主張は、独立な再実装でも支持された。

## 2. 他データセット（μ=1, 30 反復）

論文の D1–D9 のうち $(n,d)$ で同定できた 3 件（D4=Abalone, D6=Wine Quality, D8=Appliances Energy）を
`fetch_openml` で取得し、E3 と同じ手順（RelMSE・non-private bounds）で追試した（`scripts/05_other_datasets.py`）。
$(n,d)$ はいずれも論文と一致した。

![E5: other datasets RelMSE vs paper](results/figures/e5_other_datasets.png)

| データ (n, d) | 本追試 OLS | 本追試 BinAgg | 論文 OLS | 論文 BinAgg |
|---|---:|---:|---:|---:|
| Abalone (4177, 10) | 0.044 | 0.057±0.009 | 0.044 | 0.059 |
| Wine Quality (6497, 12) | 0.027 | 0.037±0.005 | 0.016 | 0.022 |
| Appliances (19735, 27) | 0.438 | 0.497±0.009 | 0.438 | 0.507 |

- **事実**: **Abalone と Appliances は論文値をほぼ再現**（OLS は完全一致、BinAgg も 0.057 vs 0.059、0.497 vs 0.507）。
  D7（Air Quality）と合わせ、3/4 のデータで論文の RelMSE を再現できた。
- **事実**: **Wine Quality は $(n,d)$ は一致するが値が約 1.7 倍**（OLS 0.027 vs 0.016 など）。原因は前処理の不一致と推定する
  （論文 Appendix が不明。red/white の結合方法・型列の扱い・スケーリングに敏感）。
- 前処理の前提（best-effort）: Abalone は Sex を 3 列 one-hot（数値 7＋3＝10）、Wine は red+white 結合に type 列を加えて 12、
  Appliances は date を除く 27 列（rv1/rv2 等を含む）。target はそれぞれ Rings / quality / Appliances。

## 3. 反復回数の影響

本体 E3 は当初 30 反復だったが、論文と同じ **100 反復**に増やしても D7 の結果は実質変わらなかった
（μ=1 で BinAgg RelMSE = 0.450±0.006、30 反復時の 0.450±0.005 とほぼ同一）。RelMSE が反復に対して安定であることを確認した。

## 4. 合成データの下流 ML 評価（論文 §5.2.2 / Table 3, ε=1）

レポート本体（E1–E3）と上の発展は、いずれも**回帰**（係数・線形予測）で有用性を測った。論文 [[1]](#ref1) の**もう一つの主眼**は
**合成データで非線形 ML を学習し実テストで予測する**評価（§5.2.2, Table 3）である。本節はこの未カバー軸を best-effort で
追試する（`scripts/07_downstream_ml.py`、計画 `docs/plans/E4-downstream-ml.md`）。手順は論文に準拠する: 各データを
**80/20** に分割し、**train 区画のみ**から BinAgg 合成データ（Algorithm 3）を生成、4 つの下流モデルを学習し**実 test** で評価
（train-on-synthetic / test-on-real）。予算は論文と同じ **ε=1, δ=1/n^1.1** を μ-GDP に換算（`mu_from_eps_delta` → μ≈0.28–0.31）。
合成データは 10 本生成して平均する。

> 📘 best-effort・逸脱の注記:
> - **指標の正規化**: 論文 Table 3 の Original 列（例 Abalone 0.487, Air Quality 0.489）の**スケール**は、E3/E5 で用いた
>   $\lVert y\rVert^2$ 正規化（Abalone で 0.044）では再現できず、**分散正規化 RelMSE$_\text{var}$ = MSE/Var$(y)$（= $1-R^2$、目的変数の標準化と等価）**で
>   両データとも論文と同スケールになる。よって本節は RelMSE$_\text{var}$ を**主指標**とし（$\lVert y\rVert^2$ 版は JSON に併記）、これを Table 3 と照合する。
> - **下流モデル**: 論文は XGBoost / RandomForest / SVR / MLP。本追試は依存追加を避け **XGBoost を sklearn の
>   HistGradientBoosting で代替**、残り 3 つは同一。SVR/MLP は学習を **5000 標本にサブサンプル**（計算量、木系は全件）、SVR/MLP のみ標準化。
> - **比較手法**: 論文の競合（AIM・DP-CTGAN・PATE-CTGAN・DPGAN・PATEGAN, SmartNoise）は重い別系統のため**未実装**で、
>   Table 3 の報告値を文脈として引用する（別 Issue 連携）。

![E4: downstream-ML utility of synthetic data vs paper Table 3](results/figures/e4_downstream_ml.png)

| データ (n, d) | Original 本追試 | Original 論文 | BinAgg 本追試 | BinAgg 論文 |
|---|---:|---:|---:|---:|
| Abalone (4177, 10) | 0.43 | 0.487 | 0.90 | 0.731 |
| Wine Quality (6497, 12) | 0.34 | 0.628 | 0.97 | 1.195 |
| Air Quality (9357, 12) | 0.46 | 0.489 | 0.86 | 0.584 |
| Appliances (19735, 27) | 0.73 | 0.683 | 1.82 | 1.490 |

*表: 下流 ML の RelMSE$_\text{var}$（4 モデル×合成 10 本の平均, ε=1）。小さいほど良い。Original=実データ学習の非プライベート参照。*

BinAgg 合成データ学習時のモデル別 RelMSE$_\text{var}$（参考）:

| データ | GradBoost(XGB代替) | RandomForest | SVR | MLP |
|---|---:|---:|---:|---:|
| Abalone | 0.87 | 0.85 | 0.73 | 1.16 |
| Wine | 0.80 | 0.75 | 0.73 | 1.59 |
| Air Quality | 0.87 | 0.88 | 0.84 | 0.85 |
| Appliances | 1.66 | 1.35 | 1.01 | 3.25 |

- **事実**: Original（非プライベート参照）の RelMSE$_\text{var}$ は **3/4 で論文と同水準**（Abalone 0.43 vs 0.487、Air Quality 0.46 vs 0.489、
  Appliances 0.73 vs 0.683）。Wine は本追試が良い（0.34 vs 0.628）が、これは E5 でも見た **Wine の前処理差**と整合する。
- **事実**: BinAgg 合成データで学習した下流モデルの RelMSE$_\text{var}$ は、**全データで Original より大きい（プライバシーが有用性を劣化させる）**が、
  **論文 BinAgg と同オーダー**に収まる（Abalone 0.90 vs 0.731、Appliances 1.82 vs 1.490 など）。本追試がやや大きめなのは、ε=1→μ≈0.3 と
  強プライバシー域であること、競合手法のような調整を行っていないこと、**MLP が外れ気味**（Appliances 3.25, Wine 1.59）で平均を押し上げることが要因と推定する。
- **考察**: 論文 §5.2.2 の核心——**BinAgg の合成データは、下流 ML でも非プライベート参照と同オーダーの実用的な予測性能を与える**——は、
  独立な再実装でも**質的に支持**された。絶対値は best-effort のため一致しないが、スケール・劣化の向き・データ間の相対関係（Air Quality が最も Original に近い等）は整合する。
  なお論文の主張「BinAgg が DP-SDG 手法中で最良（8 中 5）」の検証には競合手法の実装が必要で、本追試の範囲外（別 Issue）。

## まとめと残課題

- 比較手法（AdaSSP）・他データ（Abalone/Wine/Appliances）・反復 100・**下流 ML 評価（§5.2.2）**のいずれでも、BinAgg は論文と整合する挙動を示した。
  特に予測精度の桁・劣化の向き・データ間の相対関係を再現できた。
- 残る発展: **§5.2.2 の競合 DP-SDG 手法（AIM 等）の実装**による「BinAgg が最良」主張の検証、DP-GD の追加、
  同定困難な D1/D2/D3/D5/D9、Wine の前処理一致、係数レベル評価の安定化（[補足実験](supplementary.html)の標準化参照）。

## 参照

<ol class="refs">
<li id="ref1">Lin, S., Slavković, A., &amp; Bhoomireddy, D. R. (2026). <em>Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees.</em> AISTATS 2026 (PMLR). arXiv:2510.16974. <a href="https://arxiv.org/abs/2510.16974">https://arxiv.org/abs/2510.16974</a></li>
<li id="ref2">Wang, Y.-X. (2018). Revisiting differentially private linear regression: optimal and adaptive prediction &amp; estimation in unbounded domain. UAI 2018.（AdaSSP）</li>
<li id="ref3">UCI Machine Learning Repository — Abalone / Wine Quality / Appliances Energy Prediction / Air Quality.</li>
</ol>
