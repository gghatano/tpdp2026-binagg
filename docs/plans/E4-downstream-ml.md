# 実験E4: 合成データの下流ML評価（論文 §5.2.2 の再現, Issue #1）

対象: Lin, Slavković & Bhoomireddy (2026), AISTATS 2026 ([arXiv:2510.16974](https://arxiv.org/abs/2510.16974)) **§5.2.2 "Synthetic Data Generation"**（Table 3）。
BinAgg コード: [Shuronglin/BinAgg](https://github.com/Shuronglin/BinAgg) @ `13c09bb`。

## 論文準拠プロトコル（§5.2.2 / Table 3 から抽出）

| 項目 | 論文 §5.2.2 の設定 |
|---|---|
| データ | D1–D8（Table 3）。本追試は $(n,d)$ で同定済みの **D4 Abalone(4177,10) / D6 Wine(6497,12) / D7 AirQuality(9357,12) / D8 Appliances(19735,27)** を対象 |
| 合成データ生成 | **BinAgg Algorithm 3**（bin 単位の DP 集約から合成標本を直接サンプリング）。本リポジトリの `generate_synthetic_data`（= Option C）に対応 |
| 下流モデル | **XGBoost / Random Forest / SVR / MLP** の4種（n<500 のデータでは MLP を除外＝本対象は全て n≥4177 なので全モデル使用） |
| 主要指標 | **RelMSE = ‖ỹ − y‖₂² / ‖y‖₂²**（実テスト集合上の予測誤差。§5.2.1 と同定義） |
| train/test | 各データを **80/20** に分割。合成は train 区画から生成し、**実 test** で評価（train-on-synthetic, test-on-real） |
| 反復 | 各手法につき **合成データ10本**を生成して平均 |
| プライバシー | **単一予算 ε=1, δ=1/n^1.1**（μ-GDP を Proposition 2.1 で (ε,δ) に換算）。μ への逆換算は `mu_from_eps_delta(1.0, n^-1.1)` → 本対象では **μ≈0.28–0.31** |
| 比較手法 | SmartNoise の **AIM / DP-CTGAN / PATE-CTGAN / DPGAN / PATEGAN**（AdaSSP/DP-GD ではない） |
| 照合対象 | **Table 3**（列: Original, AIM, BinAgg, …）。BinAgg は 8 データ中 5 で DP 手法中最良。Original=非プライベート参照 |

論文 Table 3 の該当値（Original / BinAgg）:

| データ | Original | BinAgg |
|---|---:|---:|
| D4 Abalone | 0.487 | 0.731 |
| D6 Wine | 0.628 | 1.195 |
| D7 AirQuality | 0.489 | 0.584 |
| D8 Appliances | 0.683 | 1.490 |

## 逸脱ログ（論文設定との差分）

| 項目 | 論文 | 本追試 | 理由 | 本文での扱い |
|---|---|---|---|---|
| XGBoost | XGBoost | sklearn `HistGradientBoostingRegressor` | 依存追加を避けた勾配ブースティングの代替。Issue も「勾配ブースティング等」と一般化 | 補助タブで「XGBoost 代替」と明記 |
| 比較手法 | AIM 等 5 手法 | **未実装**（論文値を併記のみ） | SmartNoise の重い別系統。別Issue連携 | Original/BinAgg のみ実測、競合は論文 Table 3 を引用 |
| 前処理スケーリング | 明記なし | SVR/MLP のみ train 上で `StandardScaler`（Pipeline でリーク防止） | 距離・勾配ベース手法の標準実務。木系は不要 | 補助タブに明記 |
| 予算 | ε=1 固定 | ε=1 を μ に換算して実行（μ≈0.3） | BinAgg は μ-GDP 実装。換算は公式 `mu_from_eps_delta` 使用 | 換算式と μ 値を明記 |

> best-effort 注記: 競合手法の絶対値一致は保証しない。Original/BinAgg の RelMSE 水準と「BinAgg が Original にどれだけ近いか／劣化の桁」を論文 Table 3 と照合する目的。

## 目的・仮説

- **目的**: レポート未カバー軸（下流 ML 有用性）を埋める。E2=係数誤差・E3=線形予測 RelMSE に対し、本実験は **非線形 ML を合成データで学習し実テストで評価**する論文の主眼（§5.2.2）を再現。発展（extensions）タブに best-effort で記載。
- **仮説**: BinAgg 合成データで学習した下流モデルの RelMSE は Original（非プライベート学習）より大きいが同じ桁に収まり、論文 Table 3 の傾向（D7 で Original に近い／D6・D8 で劣化大）を質的に再現する。

## 指標

- **主指標 RelMSE**（論文 §5.2.1 定義、出典 [1]）= 実テスト上の予測二乗誤差 / ‖y_test‖²。データ単位で **4モデル×10合成データ**を平均。モデル別内訳も保存。
- **補助 R²**（Issue で言及。論文は未使用）→ 併記のみ。supplementary 扱い可。

## 比較条件・公平性

- 対象: **Original（実 train で学習）** vs **BinAgg合成（μ≈0.3）**。同一の train/test 分割（seed 固定）・同一の4モデル・同一指標。
- 合成は train 区画のみから生成（test リークなし）。scaler は train で fit。
- 反復: 合成データ seed 0..9（10本）。モデルの内部乱数は固定 `random_state=0`。
- ベースライン: Original（非プライベート）が下界参照。

## 実行

- 新スクリプト **`scripts/07_downstream_ml.py`**。データローダは E3（AirQuality, ローカルCSV）/ E5（Abalone・Wine・Appliances, `fetch_openml`）を再利用。
- 出力: `results/e4_downstream_ml.json/.csv`, `results/figures/e4_downstream_ml.png`。
- 計算量の注意: Appliances(n≈19735) で SVR/MLP が重い。まずスモークで1データ・1合成・全モデルの時間を測り、必要なら Appliances の合成 train を上限サブサンプル（明記）。

## 完了基準

- データ×手法（Original/BinAgg）の RelMSE 表（4モデル平均）と、論文 Table 3 の Original/BinAgg を並記した図表。
- モデル別 RelMSE の内訳。逸脱（GB代替・scaling・ε→μ）を明記。extensions タブに反映。
