# stock-ai-prediction compute (GitHub Actions)

株価AI予測の夜間計算を GitHub Actions へ代行させるためのリポジトリ。
計算結果 (data.json) は `publish-data` ブランチに置かれ、VM 側の引き取り役が
検証して Hugging Face の公開ページ (soradash/stock-ai-prediction) へ反映する。

## 構成

コード本体は、連携ツールの制約 (一括 push が使えない) のため、base64 化した
zip ファイルとして格納している。ワークフローが実行時に復号・展開する。
正本は VM 側の作業ツリー (`~/workspace/stock_ai_pred/`) で、zip はその複製。

- `code_backend.zip` (base64 テキスト) — 中身: `phase1/backend/` 一式
  (予測アプリのバックエンド。Phase 1/2 と同一コード、`requirements.txt` を含む)
- `code_rest.zip` (base64 テキスト) — 中身: `phase0/src/poc/` (価格取得
  プロバイダ等の Phase 0 共有モジュール)、`phase1/scripts/export_static.py`
  (全銘柄の予測+注目度の書き出し)、`gha/` 一式
  - `gha/prepare.py` — HF の非公開 artifacts リポジトリから DB スナップショットと
    モデルを取得し、artifact_uri を実行環境のパスへ書き換える
  - `gha/run_nightly.sh` — 価格取込 (最新日から14日前〜今日を自動補完) → 書き出し
- `.github/workflows/nightly-predict.yml` — 平日 19:05 JST 実行、失敗時は Issue を起票

## 必要な Secrets

- `HF_TOKEN_READ` — artifacts リポジトリ (soradash/stock-ai-prediction-artifacts)
  に対する読み取り専用トークン。書き込み権限は不要。

## 運用メモ

- 週次学習は当面 VM 側で行い、学習後に DB スナップショットとモデルを
  artifacts リポジトリへ同期する (同期が最大で1週間古くても、価格取込が
  14日分を遡って補完するため夜間実行は成立する)。
- publish-data ブランチは常に単一コミット (orphan で作り直す) のため、
  リポジトリは肥大化しない。
