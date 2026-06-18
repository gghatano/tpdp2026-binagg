# 発展実験: 比較手法と他データセット

レポート本体（E3）は論文 [[1]](#ref1) の D7（Air Quality）を予測誤差 RelMSE で再現した。ここでは
カバレッジを広げる発展実験として、(1) 比較手法 **AdaSSP** の追加、(2) 同定できた**他データセット 3 件**での
追試、(3) 反復回数の影響、を best-effort でまとめる。いずれも論文 Table 2 の報告値と並べて評価する。

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

## まとめと残課題

- 比較手法（AdaSSP）・他データ（Abalone/Wine/Appliances）・反復 100 のいずれでも、BinAgg は論文と整合する挙動を示した。
  特に予測精度の優劣と桁は再現できた。
- 残る発展: DP-GD の追加（チューニング感度・コスト）、同定困難な D1/D2/D3/D5/D9、Wine の前処理一致、
  係数レベル評価の安定化（[補足実験](supplementary.html)の標準化参照）。

## 参照

<ol class="refs">
<li id="ref1">Lin, S., Slavković, A., &amp; Bhoomireddy, D. R. (2026). <em>Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees.</em> AISTATS 2026 (PMLR). arXiv:2510.16974. <a href="https://arxiv.org/abs/2510.16974">https://arxiv.org/abs/2510.16974</a></li>
<li id="ref2">Wang, Y.-X. (2018). Revisiting differentially private linear regression: optimal and adaptive prediction &amp; estimation in unbounded domain. UAI 2018.（AdaSSP）</li>
<li id="ref3">UCI Machine Learning Repository — Abalone / Wine Quality / Appliances Energy Prediction / Air Quality.</li>
</ol>
