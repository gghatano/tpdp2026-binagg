# BinAgg 再現追試レポート: DP 線形回帰と統計的保証付き合成データ生成

**対象 (Issue #16)** — Lin, Slavković & Bhoomireddy (2026),
*Differentially Private Linear Regression and Synthetic Data Generation with
Statistical Guarantees*, AISTATS 2026 [1]。
実装: [Shuronglin/BinAgg](https://github.com/Shuronglin/BinAgg) @ `13c09bb` (2026-05-27) [2]。

実行環境: Python 3.12.12 / numpy 2.4.6 / scipy 1.17.1 / pandas 3.0.3（`requirements.txt`）。
全乱数はシード固定。本レポートの数値は `results/` の生成物に対応する。

---

## Abstract

差分プライバシー下の線形回帰について、論文 [1] が提案する手法（BinAgg）の公式実装 [2] を
固定環境で動かし、論文の中核的な主張を追試した。**結論として、主張はおおむね再現された。**
すなわち、(i) この手法の信頼区間はほぼ名目どおりの被覆を達成し、ノイズを考慮しない素朴な
区間が被覆を大きく失うのと対照的であること、(ii) 合成データに素朴に回帰すると係数が偏る一方、
この手法のバイアス補正がそれを回復すること、(iii) 実データでもプライバシー予算に応じた
精度のトレードオフが期待どおり観測されること、を確認した（詳細は本文 §3）。

残課題の方向性としては、(a) 評価データ（UCI Air Quality）に潜む欠損値表現の前処理を正す、
(b) 論文本文の図表との定量的な突き合わせ、(c) 設定（データ生成過程・標本数・実データ）を
広げた感度分析、が挙げられる（詳細は §5）。

---

## 1. 背景と手法 (出典: [1][2])

- **問題** [1]: DP 線形回帰で集約量にノイズを足すと、`SᵀWS` が真の二乗和より系統的に膨らみ
  点推定が縮小方向に偏る。さらにノイズ分散を無視した標準誤差は過小評価となり、信頼区間が
  狭すぎて被覆を失う。
- **BinAgg** [1][2]:
  1. **PrivTree** でデータ空間を private に分割（binning）。
  2. 各 bin の `count / sum(X) / sum(y)` に Gaussian ノイズを加える（μ-GDP）。既定予算配分は
     `(binning, count, sum_x, sum_y) = (1,3,3,3)`、すなわち分割 10% / 各集約 30%。
  3. **Algorithm 2**: 重み付き最小二乗にバイアス補正項 D̃ を入れ
     `β̃ = (S̃ᵀW̃S̃ − D̃)⁻¹ S̃ᵀW̃t̃`。サンドイッチ分散 `Σ̃ = M̃⁻¹H̃M̃⁻¹` から CI を構成（Thm 4.2）。
  4. **Algorithm 3**: 同じ集約から bin 単位で合成標本を生成。回帰と合成は同じノイズ実現を
     共有でき（Corollary 3.1）、**同一プライバシー予算**で両方を出せる。
- **評価方針** [1]: 合成データの良否を周辺分布の保存ではなく**回帰分析（点推定・信頼区間）の
  保証**で測る点が本手法の特徴。本追試もこの方針に従う。

> 事実と推定の区別: [n] 付きは論文・コードに基づく事実。タグ無しの「考察」は本追試の解釈。

---

## 2. 実験設定

指標・ベースライン・予算・データの詳細は `docs/plans/experiment-plan.md`。要点:

- **主指標**: 95% CI 被覆率（妥当なら ≈0.95）、係数バイアス、相対 L2 係数誤差、CI 幅。
  補助: 合成 vs 元のモーメント誤差。**周辺 TVD ではなく回帰の保証**を主軸に置く [1]。
- **ベースライン**: 非プライベート OLS（utility 上界・被覆参照）、naive（補正なし）推定量、
  合成データへの素朴 OLS。
- **プライバシー予算**: μ-GDP で μ ∈ {0.5, 1.0, 2.0, 5.0}。
- **データ**: (シミュ) X∼Uniform(0,10)³, β=[1.5,−2,0.5], y=Xβ+N(0,1), n=2000。bounds は
  **既知の生成分布から**（データ非依存）。(実データ) UCI Air Quality 同梱 CSV、目的 CO(GT)、
  12 特徴、bounds はドメイン知識（BinAgg example 準拠）。

---

## 3. 結果

### E1 — 統計的保証: 95% CI 被覆率（シミュレーション, T=300 試行/μ）

`results/e1_coverage.csv` / `figures/e1_coverage.png`

![E1: CI coverage and coefficient bias vs μ](results/figures/e1_coverage.png)

| μ | 被覆率 BinAgg(補正) | 被覆率 naive | 平均\|バイアス\| BinAgg | 平均\|バイアス\| naive | 平均 CI 幅 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | **0.917** | 0.077 | 0.064 | 0.066 | 0.518 |
| 1.0 | **0.931** | 0.149 | 0.032 | 0.032 | 0.291 |
| 2.0 | **0.934** | 0.227 | 0.019 | 0.019 | 0.181 |
| 5.0 | **0.918** | 0.318 | 0.012 | 0.012 | 0.103 |

- **事実**: BinAgg の補正版 CI は全 μ で被覆率 0.92–0.93 と公称 0.95 に近接。係数別でも
  0.89–0.97 の範囲（β₂=−2.0 が僅かに低めの ≈0.89）。
- **事実**: ノイズ分散を無視した naive CI の被覆率は 0.08–0.32 と大きく不足。CI 幅は μ とともに
  単調に縮小（0.52→0.10）。
- **考察**: 主張「妥当な信頼区間」は再現された。点推定のバイアスは本デザイン（良条件な一様計画）
  では補正版と naive でほぼ同等であり、**両者を分けるのは点推定よりも分散推定（サンドイッチ SE）**
  だと読める。0.95 をやや下回るのは asymptotic CI の有限標本近似と整合的（限界 §5）。

### E2 — 合成データの回帰 utility とバイアス補正の効果（シミュレーション, 20 シード平均）

`results/e2_synthetic_utility.csv` / `figures/e2_utility.png`

![E2: regression utility vs μ](results/figures/e2_utility.png)

相対 L2 係数誤差（小さいほど良い）:

| μ | OLS(元データ, 参照) | naive OLS(合成) | BinAgg DP 回帰 (β̃) |
|---:|---:|---:|---:|
| 0.5 | 0.003 | 0.575 | **0.093** |
| 1.0 | 0.003 | 0.327 | **0.058** |
| 2.0 | 0.003 | 0.172 | **0.031** |
| 5.0 | 0.003 | 0.064 | **0.018** |

- **事実**: 合成データへ素朴に OLS を当てると係数誤差が大きい（μ=0.5 で 0.58）。BinAgg の
  補正推定は同一データから誤差 0.09 と 1 桁近く改善し、μ↑で OLS(元) の 0.003 へ漸近する。
- **考察**: 「合成データをそのまま回帰すると偏る／補正で真値を回復する」という主眼を確認。
  utility を**回帰の保証**で測る論文の評価軸が、周辺分布の一致より厳しく実用的であることを示す。

### E3 — 実データ再現: UCI Air Quality（n=9357, d=12, 5 シード平均）

`results/e3_airquality.csv` / `figures/e3_airquality.png`。Option A/B/C をいずれも実行。

![E3: CI width and synthetic moment error vs μ](results/figures/e3_airquality.png)

| μ | 平均 95% CI 幅 | \|mean(y) 誤差\| | \|std(y) 誤差\| |
|---:|---:|---:|---:|
| 0.5 | 1.461 | 0.856 | 1.643 |
| 1.0 | 1.068 | 0.468 | 0.943 |
| 2.0 | 0.776 | 0.217 | 0.383 |
| 5.0 | 0.387 | 0.059 | 0.036 |

（モーメント誤差はクリップ後の元データ mean=1.766 / std=1.554 を基準。理由は §5。）

- **事実**: CI 幅・合成モーメント誤差はいずれも μ とともに単調に縮小。Option C で合成標本数
  9357（元と同数, `preserve_sample_size`）・使用 bin 159 を確認。Option A/B は同一予算で動作。
- **考察**: 実データでもプライバシー–精度トレードオフが教科書どおりに観測され、実装は破綻なく
  動く。一方、非プライベート OLS との係数一致は本データでは妥当な utility 指標にならない（§5）。

---

## 4. 再現手順（要約）

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt \
  "git+https://github.com/Shuronglin/BinAgg.git@13c09bb"
.venv/Scripts/python.exe scripts/01_sim_coverage.py      # E1
.venv/Scripts/python.exe scripts/02_synthetic_utility.py # E2
.venv/Scripts/python.exe scripts/03_real_airquality.py   # E3
```

各スクリプトは 1 回実行で `results/*.json|*.csv` と `figures/*.png` を生成（シード固定で再現可能）。

---

## 5. 限界と注意点

1. **欠損値センチネル `-200`（再現上の落とし穴・本追試の発見）**: UCI Air Quality は欠損を
   `-200` で表す。BinAgg 同梱 example のクリーニングはこれを除去せず、CO(GT) では 1683/9357 行が
   `-200`。生の y は mean=−34.21 / std=77.65 と汚染される。DP 経路は bounds(0,15) クリップで
   一部遮蔽されるが、**(a) 合成モーメントを生の y と比較すると無意味な巨大誤差**になり、
   **(b) 非プライベート OLS 参照自体が汚染**される（例: AH 係数 −0.895）。本追試は (a) をクリップ後
   基準で回避したが、E3 の DP–OLS 係数一致は utility 指標として信頼できない。**正しい前処理は
   `-200` 行の除去**であり、ベンチ設計上の重要点として記録する。
2. **単一の生成過程・単一実データ**: E1/E2 は 1 種類の線形生成過程、E3 は 1 データセットのみ。
   一般化には設計（特徴分布・次元・条件数・誤差分布）と複数データへの拡張が必要。
3. **漸近 CI の有限標本近似**: 被覆率が 0.95 をやや下回る（0.92–0.93）のは漸近正規近似の
   有限標本誤差と整合的。n や bin 数を増やした感度分析は未実施。
4. **論文本文との数値対応は未照合**: 本追試は実装 [2] の挙動再現であり、論文 [1] の図表との
   定量一致は確認していない（arXiv 本文の精読は Issue タスクの残作業）。
5. **bounds の与え方**: シミュは生成分布の既知範囲、実データはドメイン知識で固定。bounds を
   private に推定する経路（予算分割）は評価していない。

---

## 6. 結論

公式実装 BinAgg を固定環境で動かし、論文 [1] の中核主張を再現できた。すなわち
(i) バイアス補正＋サンドイッチ分散による **95% CI はほぼ公称被覆を達成**し、ノイズを無視する
素朴推定は被覆を失う (E1)、(ii) **合成データ上の回帰は補正なしでは偏り、BinAgg の補正が真値を
回復する** (E2)、(iii) 実データでも μ に応じた精度トレードオフが観測される (E3)。
「合成データ utility を回帰の統計的保証で測る」という論文の評価軸の有効性を、被覆率という
直接的な指標で確認した点が本追試の主な貢献である。残課題は §5 の各項（特に `-200` 前処理の
是正、論文数値との照合、設計感度分析）。

---

## References

- [1] Lin, S., Slavković, A., & Bhoomireddy, D. R. (2026). *Differentially Private
  Linear Regression and Synthetic Data Generation with Statistical Guarantees.*
  AISTATS 2026 (PMLR). arXiv:2510.16974. https://arxiv.org/abs/2510.16974
- [2] Shurong Lin et al. *BinAgg* (Python package), commit `13c09bb` (2026-05-27).
  https://github.com/Shuronglin/BinAgg
- [3] UCI Machine Learning Repository — Air Quality Data Set. (同梱 `data/AirQualityUCI.csv`)
