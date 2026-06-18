# 発展実験: 比較手法と他データセット

レポート本体（E3）は論文 [[1]](#ref1) の D7（Air Quality）を予測誤差 RelMSE で再現した。ここでは
カバレッジを広げる発展実験として、(1) 比較手法 **AdaSSP** の追加、(2) 同定できた**他データセット**での
追試（§2: Abalone/Wine/Appliances、§4: 新たに同定した D1/D2/D5/D9）、(3) 反復回数の影響、
(4) **Wine の乖離の原因切り分けと解消**（§5）を best-effort でまとめる。いずれも論文 Table 2 の報告値と並べて評価する。

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
- **事実（更新）**: 上表の **Wine Quality の乖離（OLS 0.027）は解消した**。`fetch_openml` 経由ではなく、red/white の
  **列名が揃った標準 CSV**（GitHub ミラー）で結合し直すと OLS=**0.016**・BinAgg=**0.021** となり論文値（0.016 / 0.022）に一致する。
  原因は前処理の選択ではなく **red/white の特徴量列のアラインメント**にあった（下記「4. Wine の乖離解消」）。
- 前処理の前提（best-effort）: Abalone は Sex を 3 列 one-hot（数値 7＋3＝10）、Wine は red+white 結合に type 列を加えて 12、
  Appliances は date を除く 27 列（rv1/rv2 等を含む）。target はそれぞれ Rings / quality / Appliances。

## 3. 反復回数の影響

本体 E3 は当初 30 反復だったが、論文と同じ **100 反復**に増やしても D7 の結果は実質変わらなかった
（μ=1 で BinAgg RelMSE = 0.450±0.006、30 反復時の 0.450±0.005 とほぼ同一）。RelMSE が反復に対して安定であることを確認した。

## 4. 追加同定したデータセット D1/D2/D5/D9（best-effort, μ=1）

論文の D1–D9 のうち §2 で未同定だった 5 件について、$(n,d)$ と論文の参照データから同定を進めた。
**本追試環境では OpenML と UCI が遮断**されているため、`fetch_openml` ではなく **GitHub ミラー**から取得し、
E3/E5 と同じ手順（予測 RelMSE・non-private bounds）で追試した（`scripts/07_more_datasets.py`）。

| ID | データ (n, d) | 同定 | 状態 |
|---|---|---|---|
| D1 | (182, 4) | **LT-FS-ID: Intrusion Detection in WSNs**（UCI 715, k-barriers 回帰） | 同定済・データ取得不可 |
| D2 | (345, 6) | **BUPA Liver Disorders** | 追試済 |
| D3 | (2043, 8) | — | **未同定** |
| D5 | (5875, 21) | **Parkinsons Telemonitoring** | 追試済 |
| D9 | (21263, 81) | **Superconductivity** | 追試済 |

![E7: identified D2/D5/D9 RelMSE](results/figures/e7_more_datasets.png)

| データ (n, d) | 本追試 OLS | 本追試 BinAgg (μ=1) | 反復 |
|---|---:|---:|---:|
| D2 BUPA Liver (345, 6) | 0.406 | 0.770 ± 0.320 | 30 |
| D5 Parkinsons Telemon. (5875, 21) | 0.011 | 0.015 ± 0.002 | 30 |
| D9 Superconductivity (21263, 81) | 0.131 | 0.201 ± 0.010 | 5 |

- **事実**: D2/D5/D9 とも $(n,d)$ が論文と**一致**。いずれも BinAgg（μ=1）の RelMSE は OLS の数十%増にとどまり、
  BinAgg が DP 下で予測精度を保つという論文の主張と整合する。
- **D1（LT-FS-ID, WSN）**: ユーザ提供の参照で同定。回帰タスク（面積・センシング/通信半径・センサ数の 4 特徴 →
  k-barriers 数）で $(n,d)=(182,4)$ が一致する。ただし UCI・Kaggle・著者サイトがいずれも遮断され、GitHub raw ミラーも
  見つからなかったため、本環境では RelMSE を再現できなかった（同定のみ）。
- **D3 (2043, 8)** のみ未同定（$(n,d)$ から一意に絞れず、確定には論文 Appendix のデータ表が要る。同 arXiv も本環境では遮断）。
- 前処理の前提（best-effort, 論文 Appendix 非公開のため $d$ 一致を優先した仮定）:
  - **D2 BUPA**: 7 列（mcv, alkphos, sgpt, sgot, gammagt, drinks, selector）。target=drinks、特徴は残り 6 列（5 検査値＋selector）。
    $n=345$ は小さく、PrivTree 分割のばらつきで BinAgg の分散が大きい。
  - **D5 Parkinsons**: 22 列（subject#, age, sex, test_time, motor_UPDRS, total_UPDRS, 音声 16）。target=total_UPDRS、
    特徴は残り 21 列。$d=21$ に合わせると **subject#（ID）と motor_UPDRS** を含むことになり、後者は目的変数と強相関のため
    RelMSE が小さく出る点に注意。
  - **D9 Superconductivity**: 82 列（抽出特徴 81＋critical_temp）。target=critical_temp、特徴 81（曖昧さなし）。
    81 次元の PrivTree 分割が高コスト・シード依存のため、反復は **5**（他は 30）に減らした（逸脱として明記）。

> 📘 best-effort の限界: 論文 Table 2 の D2/D5/D9 の数値は本環境（arXiv 遮断）で参照できないため、上表は本追試値と
> 一致した $(n,d)$ のみを示す。前処理仮定（特に $d$ を合わせるための target/列選択）が差異の主因になり得る。

## 5. Wine Quality の乖離解消（前処理の切り分け）

§2 で Wine（D6）のみ OLS RelMSE が論文の約 1.7 倍（0.027 vs 0.016）だった。原因を `scripts/08_wine_preprocessing.py` で
切り分けた。鍵となる事実は、**OLS の予測 RelMSE は特徴量の列ごとのスケーリングに対して不変**（$X$ の列空間と $y$ のみに依存）という点である。
したがってズレは「標準化の有無」では説明できず、**設計行列にどの列が入るか**（切片・型列）か、**結合するデータ自体**に起因する。

| 設計（n=6497） | d | OLS RelMSE |
|---|---:|---:|
| 11 化学 + type, 切片なし（E5 と同じ） | 12 | 0.0156 |
| 11 化学 + type + 切片 | 13 | 0.0155 |
| 11 化学 + 切片, type なし | 12 | 0.0156 |
| 11 化学のみ, 切片なし | 11 | 0.0156 |

- **事実**: 切片・型列の有無によらず OLS RelMSE は **約 0.0156**（≒論文 0.016）。つまり設計列の選択は原因ではない。
- **事実**: 列名の揃った標準 CSV（red/white）で結合した本実験では BinAgg も **0.021±0.003**（μ=1）で、論文 0.022 と一致する。
- **切り分け**: E5（`fetch_openml`）は red を**名前付き列**、white を `V1..V11` として「同順」と仮定して縦結合していた。
  white の特徴量列の順序を**ランダムに入れ替える**と OLS RelMSE は 0.016〜**0.031** に広がり、E5 の 0.027 はこの
  **red/white 特徴量列のミスアラインメント**で説明できる。標準 CSV（同名・同順）で結合すれば論文値に一致する。
- **結論**: Wine の乖離は前処理（標準化・型列）ではなく **データ源と列アラインメント**の問題。aligned な正準データで **0.016 / 0.021** を再現した。

## まとめと残課題

- 比較手法（AdaSSP）・他データ（Abalone/Wine/Appliances）・反復 100 のいずれでも、BinAgg は論文と整合する挙動を示した。
  特に予測精度の優劣と桁は再現できた。
- データセットの広がり（§4）: **D1=LT-FS-ID(WSN)・D2=BUPA Liver・D5=Parkinsons Telemonitoring・D9=Superconductivity**
  を新たに同定し、データを取得できた D2/D5/D9 を best-effort 追試した。**Wine の乖離は列アラインメント由来と判明し解消**（§5）。
- 残る発展: DP-GD の追加（チューニング感度・コスト）、**唯一未同定の D3 (2043, 8)** の特定、
  係数レベル評価の安定化（[補足実験](supplementary.html)の標準化参照）。

## 参照

<ol class="refs">
<li id="ref1">Lin, S., Slavković, A., &amp; Bhoomireddy, D. R. (2026). <em>Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees.</em> AISTATS 2026 (PMLR). arXiv:2510.16974. <a href="https://arxiv.org/abs/2510.16974">https://arxiv.org/abs/2510.16974</a></li>
<li id="ref2">Wang, Y.-X. (2018). Revisiting differentially private linear regression: optimal and adaptive prediction &amp; estimation in unbounded domain. UAI 2018.（AdaSSP）</li>
<li id="ref3">UCI Machine Learning Repository — Abalone / Wine Quality / Appliances Energy Prediction / Air Quality / BUPA Liver Disorders / Parkinsons Telemonitoring / Superconductivity / LT-FS-ID Intrusion Detection in WSNs (dataset 715).</li>
<li id="ref4">本環境では OpenML / UCI が遮断されているため、§4・§5 のデータは GitHub ミラーから取得した（BUPA: <code>PepeAlex/BUPA_Liver_Disorders</code>、Parkinsons: <code>pqrst/ParkinsonsDiseaseDataAnalysis</code>、Superconductivity: <code>saranggalada/ML_Superconductivity</code>、Wine: <code>zygmuntz/wine-quality</code>）。</li>
</ol>
