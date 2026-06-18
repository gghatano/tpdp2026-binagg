# エンジニアリングと再現手順

本ページは、BinAgg の再現追試リポジトリ `tpdp2026-binagg` を **clone-and-run** で再現するための実装ノートです。環境構築・実験実行・サイトビルド・既知の落とし穴を、実際に踏んだ手順そのままに記録します。コマンドは Windows 11 + git-bash を前提に動く形で示します。

## 動作環境

- OS: Windows 11
- シェル: git-bash（Unix 風のパス区切り `/` を使う）
- パッケージマネージャ: uv 0.11.6
- Python: **3.12.12 を固定**

> ⚠️ 落とし穴: システムにインストールされている Python は 3.14 ですが、numpy・scipy・pandas の wheel がまだ 3.14 を十分にカバーしておらず、ソースビルドに落ちて失敗します。`uv venv --python 3.12` で **3.12 系を明示的に固定**してください。

仮想環境は `.venv` に作成します。Windows では Python 本体は `.venv/Scripts/python.exe` に置かれます（POSIX の `.venv/bin/` ではありません）。本ページのコマンドはすべてこのパスを直接叩く形で書きます。

## セットアップ

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt \
  "git+https://github.com/Shuronglin/BinAgg.git@13c09bb"
```

固定している依存は以下のとおりです（`requirements.txt`）。

- 数値・データ処理: `numpy==2.4.6`, `scipy==1.17.1`, `pandas==3.0.3`, `scikit-learn==1.9.0`, `matplotlib==3.11.0`
- サイトビルド用: `markdown==3.10.2`, `pymdown-extensions==10.21.3`

BinAgg は再現したコミット `13c09bb`（2026-05-27）に固定してインストールします。BinAgg 本体の依存は `numpy>=1.20`, `scipy>=1.7` のみと軽量で、上で固定したバージョンに収まります [[2]](#ref2)。

> 💡 ヒント: `requirements.txt` には BinAgg 自体は書かず、フル SHA `13c09bb3e32f9a2c2915df3599d99b79a057b05d` をコメントで残しています。短縮 `13c09bb` でもインストールできますが、厳密に再現したい場合はフル SHA を使ってください。

## 実験の実行

3 本のスクリプトを順に実行します。いずれも引数なしで動きます。

```bash
.venv/Scripts/python.exe scripts/01_sim_coverage.py      # E1 CI被覆率
.venv/Scripts/python.exe scripts/02_synthetic_utility.py # E2 合成データの回帰有用性
.venv/Scripts/python.exe scripts/03_real_airquality.py   # E3 実データ(Air Quality)
```

各スクリプトは 1 回の実行で `results/*.json` / `results/*.csv` と `results/figures/*.png` を生成します。全乱数はシード固定で、同じ環境であれば結果は再現します。

所要時間の目安（参考値）:

| 実験 | 内容 | 目安 |
| --- | --- | --- |
| E1 | CI 被覆率シミュレーション | 約 16 秒 |
| E2 | 合成データの回帰有用性 | 約 3 秒 |
| E3 | 実データ（Air Quality） | 約 9 秒 |

> 💡 ヒント: 共通処理（シード設定・出力先の作成・図の保存など）は `scripts/common.py` にまとまっています。新しい実験を足すときはここを再利用してください。

## サイトのビルドと公開

`content/*.md`（手書き Markdown）を `htmls/*.html`（自己完結 HTML）へ変換するビルダーが `scripts/03_build_html.py` です。

```bash
uv pip install --python .venv markdown pymdown-extensions
.venv/Scripts/python.exe scripts/03_build_html.py
```

ビルダーの設計上のポイント:

- **図は base64 埋め込み**: 生成 HTML は単体で共有できる自己完結ファイルになります。
- **数式は MathJax + `pymdownx.arithmatex`**（`generic: True`）でレンダリングします。
- **ページ構成はビルダー内の `PAGES` テーブルが単一の真実 (single source of truth)**。上部タブナビと目次サイドバー（h2/h3 から自動生成）はここから注入されます。ページを追加・改名するときは `PAGES` を編集します。

ビルドは Markdown→HTML 変換のみなので数十秒で終わります。実験を回し直す必要はありません。

公開は GitHub Actions（`.github/workflows/deploy-pages.yml`）が `htmls/` を GitHub Pages へデプロイします。Pages 側の build_type は **workflow** に設定します（ブランチ公開ではありません）。

> ⚠️ 落とし穴: `htmls/` はビルド成果物です。**直接編集しないでください**。手で直しても次のビルドで上書きされます。本文の修正は必ず `content/*.md` 側で行います。

## 落とし穴と対処

実際に踏んだものを「問題 → 詳細 → 対処」で残します。

### スモークテストを先に通す

- 問題: 重い実験をいきなり回すと、binagg のインストール不備に最後まで気づけない。
- 詳細: 環境構築直後は import や基本動作が壊れていることがある。
- 対処: 本実験の前に **BinAgg 同梱の example を 1 本動かして** import と基本動作を確認し、緑になってから E1〜E3 に進む。

### Windows シェルのパス

- 問題: POSIX 流のコマンドがそのまま動かない。
- 詳細: venv のバイナリは `bin/` ではなく `Scripts/` 配下。
- 対処: パスは `/` 区切りで書き、Python は `.venv/Scripts/python.exe` を直接呼ぶ。出力捨て先は `NUL` ではなく **`/dev/null`** を使う（git-bash 前提）。

### 改行コードの警告

- 問題: `git add` 時に `LF will be replaced by CRLF` が出る。
- 詳細: Windows の自動改行変換による警告。
- 対処: **無害**。そのまま進めてよい。

### GitHub Actions の Node 警告

- 問題: ログに `Node.js 20 is deprecated` が出る。
- 詳細: 一部アクションのランタイム表記に由来する警告。各アクションは実際には Node 24 で実行され成功している。
- 対処: **無害**。デプロイ結果が success なら問題なし。

### MathJax の表示数式がリスト内で崩れる

- 問題: `$$...$$` の表示数式が変換されず、生の `$` が残る。
- 詳細: リスト項目のインデント下に置くと Markdown がブロック数式として認識しない。
- 対処: 表示数式は**独立行に、前後を空行で囲んで**書く。リストの中に押し込まない。

### Air Quality の欠損値 `-200`

- 問題: 実データで一部の値が `-200` になっている。
- 詳細: Air Quality データセットでは欠損が `-200` というセンチネル値で表現されており、そのまま数値計算すると結果が壊れる前処理上の罠。
- 対処: 前処理で `-200` を欠損として扱う。詳細は「前処理とドメイン特化メモ」ページを参照。

## リポジトリ構成

```
tpdp2026-binagg/
├── content/            # 手書きの Markdown（編集する本文）
├── htmls/              # ビルド成果物（直接編集しない）
├── scripts/            # 実験スクリプト + サイトビルダー
│   ├── common.py
│   ├── 01_sim_coverage.py
│   ├── 02_synthetic_utility.py
│   ├── 03_real_airquality.py
│   └── 03_build_html.py   # content/*.md -> htmls/*.html（PAGES が単一の真実）
├── data/               # 入力データ
├── results/            # 実験出力（json/csv, figures/*.png）
├── requirements.txt    # 固定依存
└── .github/workflows/  # deploy-pages.yml（Pages へデプロイ）
```

## 参照

<ol class="refs">
<li id="ref2">BinAgg (Python package). Shuronglin/BinAgg, commit 13c09bb (2026-05-27). <a href="https://github.com/Shuronglin/BinAgg">https://github.com/Shuronglin/BinAgg</a></li>
</ol>
