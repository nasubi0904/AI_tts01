# orun

Ollama の `ollama run` に挙動を寄せた軽量 Python CLI です。`/api/chat` を既定で利用し、会話履歴・ストリーミング・seed 固定などの比較検証を行えます。

## インストールと実行

`requests` が利用できる Python 3.9+ 環境で以下を実行してください。

```bash
pip install requests  # 未導入の場合
python orun.py <model> "<prompt>"
```

## 主な使い方

- ワンショット: `python orun.py gpt-oss:20b "こんにちは。自己紹介を1行で。"`
- REPL: `python orun.py gpt-oss:20b`
  - `> 2+2=?`
  - `> /reset`
  - `> では3+5は？`
- seed 固定比較: `python orun.py gpt-oss:20b "2+2=?" --seed 1 --temperature 0 --no-stream`
- 生成 API: `python orun.py gpt-oss:20b "要約して" --generate`
- JSON 出力: `python orun.py gpt-oss:20b "短く返答" --json`

## 主なオプション

- `--host` (既定: `http://127.0.0.1:11434`)
- `--system "<text>"` : system プロンプトを先頭に付与
- `--seed <int>` : 乱数シード
- `--temperature <float>` (既定: `0`)
- `--top-p <float>` (既定: `1.0`)
- `--num-predict <int>` : 生成トークン数上限
- `--keep-alive <str>` (既定: `30m`)
- `--no-stream` : ストリーミングを無効化
- `--json` : チャンク/最終応答を JSON 行で出力
- `--show-payload` : 送信 JSON を stderr に表示
- `--generate` : `/api/generate` を利用するワンショットモード

## 注意事項

- Ollama サーバーが停止中・ポート違いの場合は接続エラーになります。ホスト設定を確認してください。
- `/reset` コマンドで会話履歴のみリセットします (system プロンプトは保持)。
- ストリーミング無効化時は応答完了後にまとめて表示されます。

MIT License
