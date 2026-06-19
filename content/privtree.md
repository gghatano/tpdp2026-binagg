# PrivTree とは — DP な空間分割アルゴリズム

> 📘 出典: 本ページは PrivTree 原論文 [[1]](#ref1)、BinAgg 原論文 [[2]](#ref2)、BinAgg 実装 [[3]](#ref3)
> （`binagg/binning.py`）に基づく。BinAgg 全体での位置づけは「[BinAgg 手法詳細](method-binagg.html)」を参照。

BinAgg は「データ空間を **bin**（小領域）に分割してから、bin 単位の集約量にノイズを足す」二段構えを取る
（[手法詳細 §2](method-binagg.html)）。このとき、**分割の境界そのものをデータから決める**と、境界の位置が個々のレコードの
有無を漏らしてしまう。そこで BinAgg は分割自体を差分プライバシー (DP) の下で行う。その分割アルゴリズムが **PrivTree** [[1]](#ref1) である。

---

## 1. 何を解く道具か（一言で）

PrivTree は、**データ空間を再帰的に分割した木（階層的分割）を、DP を満たしながら作る**汎用アルゴリズムである [[1]](#ref1)。
密度の高い領域は細かく、疎な領域は粗く分割する。ヒストグラム・空間索引・range query など「データ密度に応じた適応的グリッド」を
DP で作りたい場面で広く使える。BinAgg では、得られた各 bin（葉ノードの矩形領域）ごとに $\mathrm{count}$・$\mathrm{sum}(X)$・
$\mathrm{sum}(y)$ を集約し、回帰と合成データ生成に用いる [[2]](#ref2)。

> 💡 直感: 一様な格子で切ると、点の少ない領域は無駄に細かく、点の多い領域は粗すぎる。PrivTree は「点が多い箱だけさらに割る」を
> DP の下で安全に繰り返し、データの形に沿った分割を得る。

---

## 2. なぜ素朴な階層分割では困るのか

「各ノードで点数を数え、多ければさらに分割する」を素朴に DP 化しようとすると、**木の深さ（レベル数）だけ繰り返し count を
公開する**ことになる。DP の合成則では、$h$ 段の分割に予算を配ると各段は $\varepsilon/h$ しか使えず、深い木ほど 1 段あたりの
ノイズが大きくなって分割がガタガタになる。木の高さを事前に決め打ちする必要も生じる。

PrivTree の核心的な貢献は、**総プライバシー予算が木の深さに依存しない**点である [[1]](#ref1)。深さを増やしても $\varepsilon$ を
分割し直す必要がなく、適応的にいくらでも深く割れる。これを 2 つの仕掛けで実現する（§3）。

---

## 3. アルゴリズム（BinAgg 実装に即して）

分割対象の領域をキューに入れ、取り出すたびに「割るか・葉にするか」を判定する幅優先の手続きである [[3]](#ref3)。
分岐数 $\beta$（既定 $\beta=2$ の二分割）と分割予算 $\varepsilon$（$\mu$-GDP の binning 予算 $\mu_{\text{bin}}$ を
$\varepsilon$ に換算 [[2]](#ref2)）から、次の 2 定数を決める。

$$
\lambda = \frac{2\beta - 1}{(\beta - 1)\,\varepsilon},
\qquad
\delta = \lambda \,\ln\beta .
$$

各ノード（領域）について、含まれる点数 $n_v$ と深さ $\ell$ から**スコア**を作り、ノイズを加えて閾値 $\theta$ と比較する。

$$
\underbrace{b_v = n_v - \delta\,\ell}_{\text{深さペナルティ}},
\qquad
\underbrace{\hat b_v = \max\!\bigl(b_v,\ \theta - \delta\bigr)}_{\text{下からの打ち切り}},
\qquad
\tilde b_v = \hat b_v + \mathrm{Laplace}(\lambda).
$$

- $\tilde b_v > \theta$ なら**分割**する。実装は最も幅の広い次元を選び、その中点で 2 分割して子ノードをキューに戻す [[3]](#ref3)。
- そうでなければ**葉**として確定し、その領域を 1 つの bin にする。

判定後、葉が少なすぎる場合は `min_bins`（既定 $d+1$）に達するまで最大の bin を強制分割する（回帰に必要な最小限の解像度を確保する後処理）[[3]](#ref3)。

### 3.1 2 つの仕掛けがなぜ「深さ非依存」を生むか

- **深さペナルティ** $-\delta\ell$: 深いノードほどスコアを下げ、際限ない深掘りを抑える。$\delta=\lambda\ln\beta$ という値は、
  分岐で増える子ノード数（$\beta$ 倍）の効果をちょうど相殺するよう設計されている [[1]](#ref1)。
- **下からの打ち切り** $\hat b_v=\max(b_v,\ \theta-\delta)$: スコアを $\theta-\delta$ 未満にしないことで、1 レコードの増減が
  経路上の全ノードのスコアに与える影響（感度）を**深さによらず定数**に抑える。

この 2 つにより、木全体での隣接データセット間のスコア変化の総量が深さに依存せず有界になり、**各ノードに $\lambda$ の
Laplace ノイズを 1 回足すだけ**で木全体が $\varepsilon$-DP を満たす [[1]](#ref1)。「段ごとに予算を割る」発想からの脱却がポイントである。

> 💡 直感: 深く割るほどスコアを少しずつ引き、かつ下限で頭打ちにする——この「減衰＋打ち切り」のおかげで、何段割っても
> 「1 人が結果に効く量」が増えない。だから深さで予算を割らずに済む。

### 3.2 パラメータの意味

| 記号 | 実装名 | 意味 | 既定/与え方 |
|---|---|---|---:|
| $\beta$ | `branching_factor` | 分岐数（2 で二分割） | 2 |
| $\theta$ | `theta` | 分割の閾値。小さい（負）ほど積極的に割る | 0 |
| $\varepsilon$ | — | 分割予算（$\mu_{\text{bin}}$ から換算） | $\mu$ 配分の約 10% |
| $\lambda$ | `lambda_` | Laplace ノイズ尺度 $=(2\beta{-}1)/((\beta{-}1)\varepsilon)$ | 上式 |
| $\delta$ | `delta_depth` | 深さペナルティ $=\lambda\ln\beta$ | 上式 |
| — | `min_bins` | 最小 bin 数（不足なら強制分割） | $d+1$ |

> ⚠️ 注意: 分割は DP だが、分割対象の**領域の外枠**（各特徴の上下限 `x_bounds`）はデータから計算してはならない。
> データ依存の境界は漏えい経路になる（[手法詳細](method-binagg.html)・[前処理・データ](data-notes.html)）。本追試の実データ評価が
> non-private bounds で「DP ではない」と明記しているのはこのためである。

---

## 4. BinAgg における役割と本追試での観察

PrivTree は BinAgg のパイプライン先頭（Algorithm 1 の binning ステップ）に位置し、総予算 $\mu$ のうち分割に約 10% を割く
（残り 90% は count・$\mathrm{sum}_x$・$\mathrm{sum}_y$ の集約に配分）[[2]](#ref2)（[手法詳細 §2.2](method-binagg.html)）。

- **$\mu$ を上げる**と分割予算も増えてノイズ $\lambda$ が小さくなり、より細かい bin が得られる（解像度↑、推定精度↑）。
- **次元 $d$ が大きいと高コスト**: 本追試では D9 Superconductivity（$d=81$）の PrivTree 分割が高コストかつシード依存だったため、
  反復回数を減らす逸脱を行った（[発展実験 §4](extensions.html)）。
- **小標本では分割が不安定**: D2 BUPA（$n=345$）では PrivTree 分割のばらつきが BinAgg の分散を押し上げた（[発展実験 §4](extensions.html)）。

> 💡 まとめ: PrivTree は「データの密度に沿った DP な bin を、木の深さに予算を取られずに作る」部品である。BinAgg の
> バイアス補正やサンドイッチ分散（[手法詳細 §3](method-binagg.html)）は、この bin 集約にノイズが乗ることを前提に組まれている。

---

## 参照

<ol class="refs">
<li id="ref1">Zhang, J., Xiao, X., &amp; Xie, X. (2016). <em>PrivTree: A Differentially Private Algorithm for Hierarchical Decompositions.</em> ACM SIGMOD 2016, pp. 155–170. <a href="https://doi.org/10.1145/2882903.2882928">https://doi.org/10.1145/2882903.2882928</a></li>
<li id="ref2">Lin, S., Slavković, A., &amp; Bhoomireddy, D. R. (2026). <em>Differentially Private Linear Regression and Synthetic Data Generation with Statistical Guarantees.</em> AISTATS 2026 (PMLR). arXiv:2510.16974. <a href="https://arxiv.org/abs/2510.16974">https://arxiv.org/abs/2510.16974</a></li>
<li id="ref3">BinAgg (Python package). Shuronglin/BinAgg, commit 13c09bb (2026-05-27)。本ページのアルゴリズム記述は <code>binagg/binning.py</code> の <code>privtree_binning</code> に対応。<a href="https://github.com/Shuronglin/BinAgg">https://github.com/Shuronglin/BinAgg</a></li>
</ol>
