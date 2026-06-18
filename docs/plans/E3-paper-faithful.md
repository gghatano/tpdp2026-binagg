# 実験 E3（論文準拠版）: UCI Air Quality でのDP線形回帰の予測誤差

`experiment-plan` §0 に従い、論文の実験セクションを一次情報として抽出して設計する。

## 論文準拠プロトコル（一次情報＝論文 [1] §5.2.1, Table 2）

| 項目 | 論文の設定（D7=Air Quality） |
|---|---|
| データ | UCI Air Quality。**$n=9357$, $d=12$**（`;`区切り・カンマ小数を変換し `to_numeric`+`dropna` のみ。欠損センチネル `-200` は残す／全特徴を使う） |
| 前処理 | 上記のみ（`-200` 除去や列削除はしない） |
| 主要指標 | **RelMSE** $=\lVert\tilde y-y\rVert_2^2/\lVert y\rVert_2^2$（$\tilde y$=予測値）。予測誤差であり係数誤差ではない |
| baseline | 非プライベート OLS。比較手法は BinAgg / AdaSSP / DP-GD |
| 予算 | μ-GDP, **μ=1**（Table 2）。θ=0 |
| bounds | **non-private bounds**（DPでないが比較目的で許容。データ由来の min/max を使用） |
| 反復 | 100 repetitions |
| 照合値（D7） | OLS **0.441** / BinAgg **0.463** / AdaSSP 0.682 / DP-GD 0.852 |

## 再現確認（着手前スモーク）

本リポジトリで n=9357・d=12・RelMSE・bounds=データ min/max・μ=1 を実装したところ:
**OLS RelMSE = 0.441（論文と完全一致）、BinAgg RelMSE ≈ 0.45（論文 0.463 と一致）**。
→ データ・指標・bounds 方針は論文準拠で正しいと確認。bounds をドメイン知識の緩い値にすると
BinAgg ≈ 1.0 に悪化する（感度過大）ため、**bounds=データ min/max** が論文準拠の鍵。

## 逸脱ログ（論文設定との差分）

| 項目 | 論文 | 本追試 | 理由 / 本文での扱い |
|---|---|---|---|
| 比較手法 | BinAgg/AdaSSP/DP-GD | **BinAgg と OLS のみ** | AdaSSP・DP-GD は BinAgg パッケージ外。対象外と明記し、論文の報告値を引用して並べる |
| 反復回数 | 100 | **30**（高速のため） | mean±std を併記。μ=1 を論文値と照合 |
| 予算 | μ=1 中心 | μ∈{0.5,1,2,5} も掃引 | トレードオフ確認。μ=1 を Table 2 と照合 |

## 評価設計

- 主指標: RelMSE（μ別、30 シード平均±std）。OLS を参照線、μ=1 を論文 D7（0.441/0.463）と並べる。
- 補助（逸脱・明示）: `-200` 除去＋NMHC(GT) 除外のクリーン版（$n=6941,d=11$）の RelMSE を**ロバストネス比較**
  として併記し、前処理の影響を示す（論文設定ではないことを明記）。
- 出力: `results/e3_airquality.json|csv`、`figures/e3_airquality.png`。

## 完了基準

- μ=1 の BinAgg RelMSE が論文 D7 値（0.463）と整合することを、並記表で示す。
- 逸脱ログがレポート本文（§前処理・§E3）に反映されている。
