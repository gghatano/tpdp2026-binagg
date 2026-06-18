# 実験計画: BinAgg 再現追試 (Issue #16)

対象論文: Lin, Slavković & Bhoomireddy (2026), *Differentially Private Linear
Regression and Synthetic Data Generation with Statistical Guarantees*, AISTATS 2026
([arXiv:2510.16974](https://arxiv.org/abs/2510.16974))。
コード: [Shuronglin/BinAgg](https://github.com/Shuronglin/BinAgg) @ `13c09bb` (2026-05-27)。

## 0. 手法の要点（再現対象）

- **Binning-Aggregation (BinAgg)**: PrivTree でデータ空間を private に分割し、各 bin の
  count / sum(X) / sum(y) に Gaussian ノイズ（μ-GDP）を加える。予算配分の既定は
  `(binning, count, sum_x, sum_y) = (1,3,3,3)`。
- **DP 線形回帰 (Algorithm 2)**: ノイズ入り集約から重み付き最小二乗を組み、
  ノイズが `SᵀWS` に持ち込むバイアスを補正項 `D̃` で除く（β̃ = (S̃ᵀW̃S̃ − D̃)⁻¹S̃ᵀW̃t̃）。
  サンドイッチ分散推定で**漸近的に妥当な信頼区間**を与える（Theorem 4.2）。
- **合成データ生成 (Algorithm 3)**: 同じ集約から bin 単位で合成標本を生成。utility は
  「周辺分布の保存」ではなく「**回帰分析（点推定・信頼区間）の保証**」で測るのが論文の主眼。

## 1. 評価指標（出典付き）

| 指標 | 定義 | 出典/根拠 |
|---|---|---|
| **CI 被覆率 (coverage)** | 真の β を 95% CI が含む試行割合。妥当なら ≈ 0.95 | 論文の主張「valid confidence intervals」(Thm 4.2) |
| **係数バイアス** | E[β̂] − β（補正あり β̃ vs 補正なし naive） | バイアス補正の有効性 (Thm 4.2) |
| **係数誤差** | ‖β̂ − β‖₂ / ‖β‖₂ | 回帰 utility |
| **CI 幅** | 上限−下限の平均 | プライバシー予算 μ と精度の関係 |
| **モーメント誤差** | 合成 vs 元データの mean/std の差 | 合成データの素性確認（補助指標） |

> 周辺分布の TVD ではなく**回帰の保証**を主指標に置く（論文の評価方針に準拠）。

## 2. ベースライン

- **非プライベート OLS**: utility の上界・被覆の参照（`numpy.linalg.lstsq`）。
- **naive（バイアス未補正）推定量**: `DPRegressionResult.naive_coefficients`。
  補正項 `D̃` の価値を示す対照。
- 「最単純合成」対照: 合成データへ**素朴に OLS** を当てた係数（バイアスが出ることを示す）。

## 3. プライバシー予算

μ-GDP: **μ ∈ {0.5, 1.0, 2.0, 5.0}**（README の強/中/弱の区分に対応）。

## 4. 実験一覧

### E1 — 統計的保証（シミュレーションデータ・CI 被覆率）
- 設定: X ~ Uniform(0,10)^3, β=[1.5,−2.0,0.5], y=Xβ+N(0,1), n=2000。bounds は**生成分布の既知範囲**から設定（データ非依存）: x_bounds=[(0,10)]×3, y_bounds=(−30,30)。
- 手順: 各試行で新規データを生成 → `dp_linear_regression(random_state=t)` → 95% CI が真値を含むか記録。T=300 試行 × μ 4 水準。
- 出力: μ 別の被覆率（補正 β̃ vs naive）、平均バイアス、平均 CI 幅。OLS 被覆を参照に併記。
- 目的: 論文の主張「妥当な信頼区間」を被覆率で定量検証。

### E2 — 合成データの回帰 utility（バイアス補正の効果）
- 設定: E1 と同じ生成過程、n=2000。
- 比較: 真値 / OLS(元) / **naive OLS(合成)** / **BinAgg DP 回帰 β̃**。係数誤差を μ 別・複数シードで平均±標準偏差。
- 目的: 「合成データへ素朴に回帰すると偏る／BinAgg の補正が真値を回復する」を示す。

### E3 — 実データ再現（UCI Air Quality）
- データ: 同梱 `data/AirQualityUCI.csv`（CO(GT) を目的、12 特徴）。bounds はドメイン知識（example 準拠）。
- 手順: Option A(回帰のみ)/B(回帰+合成・予算共有)/C(合成のみ) を μ 別に実行。DP 係数・CI 幅を OLS と比較。合成 vs 元のモーメント誤差を併記。
- 目的: 実データでの挙動再現と、μ による精度トレードオフの確認。

## 5. 再現性プロトコル

- 固定環境（`requirements.txt`、Python 3.12.12、uv）。全乱数に `random_state`/seed 固定。
- 調整は本計画段階のみ。各スクリプトは 1 回実行で `results/` に JSON+CSV と図を出力。
- 結果は上書きせず追記管理。実行条件（commit・seed・μ・n・T）を成果物に明記。
- 限界: 単一データ生成過程／単一実データ／asymptotic CI（有限標本では近似）を正直に記す。
