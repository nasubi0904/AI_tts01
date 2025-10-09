#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ai_talk_test_v4.py
VSCodeからの単体テスト。CLI可視化強化版。

ローカルで簡単に推論パラメータを編集できるように、OLLAMA の payload / options を
スクリプト内で上書きできるテンプレートを用意する。
"""
import argparse, json, os, time

# ---- 先に引数を読み、環境変数を上書きしてから ai_talk を import ----
ap = argparse.ArgumentParser(description="ai_talk v4 単体テスト")
ap.add_argument("--mode", choices=["tts","pipeline","asr"], default="pipeline")
ap.add_argument("--text", default="")
ap.add_argument("--prompt", default="")
ap.add_argument("--vosk-model", default="")
ap.add_argument("--model", default="", help="Ollamaモデル名を一時上書き")
ap.add_argument("--host", default="", help="OllamaホストURLを一時上書き")
ap.add_argument("--options", default="", help="OLLAMA_OPTIONS_JSON を一時上書き (JSON文字列)")
ap.add_argument("--payload", default="", help="OLLAMA_PAYLOAD_JSON を一時上書き (JSON文字列)")
ap.add_argument("--endpoint", default="", help="/api/generate 以外を利用したい場合のエンドポイントURL/パス")
ap.add_argument("--max-sentences", type=int, default=None, help="LLM応答の文数上限 (0で無制限)")
ap.add_argument("--max-chars", type=int, default=None, help="LLM応答の文字数上限 (0で無制限)")
ap.add_argument("--llm-timeout", type=float, default=None, help="ストリーミング応答を打ち切る秒数 (0で無制限)")
ap.add_argument("--request-timeout", type=float, default=None, help="HTTPリクエスト全体のタイムアウト秒数 (0で無制限)")
ap.add_argument("--quiet", action="store_true", help="ログ最小化")
ap.add_argument("--no-color", action="store_true", help="ANSI色を無効化")
ap.add_argument("--inspect", action="store_true", help="Ollamaサーバー情報を取得して表示する")
args = ap.parse_args()

# --- VSCodeなどから直接編集しやすいように、ここにデフォルトのpayload/optionsを用意する ---
#
# LOCAL_OLLAMA_OPTIONS / LOCAL_OLLAMA_PAYLOAD はそれぞれ OLLAMA_OPTIONS_JSON /
# OLLAMA_PAYLOAD_JSON 環境変数と同じ形式の辞書。テンプレートを差し替えるだけで
# Ollama へ渡すpayloadを細かく調整できる。
#
# ● よく使う options キーと編集例
#   - num_predict: 応答トークン数の上限。短い応答で良い場合は 128 など小さくすると待ち時間が減る。
#       LOCAL_OLLAMA_OPTIONS = {"num_predict": 128}
#   - temperature / top_p / top_k: 創造性やゆらぎを制御。温度を下げると応答が安定。
#       LOCAL_OLLAMA_OPTIONS = {"temperature": 0.6, "top_p": 0.9, "top_k": 50}
#   - repeat_penalty / presence_penalty: 繰り返しを避けたいときに使用。
#       LOCAL_OLLAMA_OPTIONS = {"repeat_penalty": 1.1}
#
# ● payload キーの追加例 (公式ドキュメント準拠)
#   - keep_alive: モデルをメモリ上に保持する時間。初回推論の待ち時間を削減。
#   - format: "json" を指定すると構造化応答が得られる。
#   - context: 直前会話のコンテキストID。会話継続時の高速化に有効。
#   - images: 画像入力を伴うモデル向けのバイナリ列 (Base64)。
#   - raw / template: プロンプトテンプレートを直接制御したい場合に使用。
#   - endpoint: OLLAMA_GENERATE_PATH 環境変数で /api/chat など別エンドポイントを指すことも可能。
#     /api/chat を選ぶと自動的に messages 形式へ変換され、system/user ロールを組み立てる。
#   - messages: 既に過去対話の履歴があればここに格納可能。末尾に今回の user 発話を自動追加する。
#
#   LOCAL_OLLAMA_PAYLOAD = {
#       "keep_alive": "10m",  # モデルを10分間キャッシュしてウォームスタート化
#       "format": "json",     # JSON出力に切り替え
#       "raw": False,          # FalseならOllama既定テンプレートを利用
#       "options": {
#           "temperature": 0.5,
#           "num_predict": 256
#       }
#   }
#
# ● 優先度の考え方
#   1. コマンドライン引数 (--options / --payload / --endpoint)
#   2. ここで定義した LOCAL_OLLAMA_*
#   3. .env / tts.env に記載した環境変数
#   すべてJSONとして解釈され、辞書でなければ自動的に空辞書に戻るため安全に試行錯誤できる。
#
# ● 応答長の制御 (num_predict を極端に下げずに待ち時間を短縮したい場合)
#   - LOCAL_STREAM_LIMITS["OLLAMA_STREAM_SENTENCE_LIMIT"] = 2  # 2文で打ち切り
#   - LOCAL_STREAM_LIMITS["OLLAMA_STREAM_CHAR_LIMIT"] = 240    # 240文字でトリミング
#   - LOCAL_STREAM_LIMITS["OLLAMA_STREAM_TIMEOUT"] = 6.0       # 6秒経過でストリーム終了
#   - LOCAL_STREAM_LIMITS["OLLAMA_REQUEST_TIMEOUT"] = 90.0     # HTTPリクエスト自体のタイムアウト
#   → これらは CLI 引数 (--max-sentences 等) が優先され、その次に LOCAL_STREAM_LIMITS、最後に .env が適用される。
#
# ● 追加の診断機能
#   - `--inspect` フラグを付けて起動すると、バージョンや利用可能モデルの一覧を取得しログ表示する。
#   - pipeline モード以外でも `--inspect` を指定すれば、実行前に接続確認だけを行える。
#
# ※ JSON 文字列化の際には ensure_ascii=False を指定しているため、日本語コメントを含めても文字化けしない。
LOCAL_OLLAMA_OPTIONS = {}
LOCAL_OLLAMA_PAYLOAD = {}
LOCAL_OLLAMA_ENDPOINT = ""
LOCAL_STREAM_LIMITS = {
    "OLLAMA_STREAM_SENTENCE_LIMIT": None,
    "OLLAMA_STREAM_CHAR_LIMIT": None,
    "OLLAMA_STREAM_TIMEOUT": None,
    "OLLAMA_REQUEST_TIMEOUT": None,
}

# LOCAL_OLLAMA_ENDPOINT に "/api/chat" を代入すると、chat スキーマを利用した推論に切り替えられる。
# 既定値の空文字列であれば .env 側の設定 (もしくは /api/generate) が使われる。
# 例: LOCAL_OLLAMA_ENDPOINT = "/api/chat"

# 以前のバージョンのスクリプトから import された場合でも AttributeError にならないよう、
# getattr で安全に取得しておく。旧版では --endpoint が存在しないため空文字に倒す。
endpoint_arg = getattr(args, "endpoint", "")

if LOCAL_OLLAMA_OPTIONS and not args.options:
    # 環境変数を直接上書きすることで ai_talk.config の初期化ロジックに乗せる。
    os.environ["OLLAMA_OPTIONS_JSON"] = json.dumps(LOCAL_OLLAMA_OPTIONS, ensure_ascii=False)
if LOCAL_OLLAMA_PAYLOAD and not args.payload:
    # payload 側も同様。options キーを含む場合は llm_client._build_payload がマージする。
    os.environ["OLLAMA_PAYLOAD_JSON"] = json.dumps(LOCAL_OLLAMA_PAYLOAD, ensure_ascii=False)
if LOCAL_OLLAMA_ENDPOINT and not endpoint_arg:
    # CLIで --endpoint を指定しなかった場合に限り、ここでの簡易指定を優先する。
    # VSCode から /api/chat と /api/generate を頻繁に切り替えたいケースに対応。
    os.environ["OLLAMA_GENERATE_PATH"] = LOCAL_OLLAMA_ENDPOINT

if args.model:
    # 一時的に別モデルを試したい時のための上書き。--options 等よりも優先度が低い。
    os.environ["OLLAMA_MODEL"] = args.model
if args.host:
    os.environ["OLLAMA_HOST"] = args.host
if endpoint_arg:
    # エンドポイントはフルURL/相対パスいずれも指定可能。llm_client 側で正規化する。
    os.environ["OLLAMA_GENERATE_PATH"] = endpoint_arg
if args.options:
    # CLIから渡したJSON文字列をそのまま保存。エラー時は config 側で空辞書化される。
    os.environ["OLLAMA_OPTIONS_JSON"] = args.options
if args.payload:
    # payload は official schema をベースに自由に拡張できる。
    os.environ["OLLAMA_PAYLOAD_JSON"] = args.payload
if args.no_color:
    os.environ["NO_COLOR"] = "1"


def _apply_limit_env(key: str, cli_value, *, is_float: bool):
    """CLI引数→ローカル設定→.env の順で応答制御用の閾値を適用する。"""

    if cli_value is not None:
        if is_float:
            value = max(float(cli_value), 0.0)
            os.environ[key] = f"{value}"
        else:
            value = max(int(cli_value), 0)
            os.environ[key] = str(value)
        return
    local_value = LOCAL_STREAM_LIMITS.get(key)
    if local_value is None:
        return
    if is_float:
        value = max(float(local_value), 0.0)
        os.environ[key] = f"{value}"
    else:
        value = max(int(local_value), 0)
        os.environ[key] = str(value)


_apply_limit_env("OLLAMA_STREAM_SENTENCE_LIMIT", args.max_sentences, is_float=False)
_apply_limit_env("OLLAMA_STREAM_CHAR_LIMIT", args.max_chars, is_float=False)
_apply_limit_env("OLLAMA_STREAM_TIMEOUT", args.llm_timeout, is_float=True)
_apply_limit_env("OLLAMA_REQUEST_TIMEOUT", args.request_timeout, is_float=True)

from ai_talk.config import VOICEVOX_URL, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_GENERATE_PATH
from ai_talk.llm_client import describe_server
from ai_talk.logger import setup, log, Reporter

def _ping(url, path):
    import requests
    try:
        r = requests.get(f"{url}{path}", timeout=3)
        return r.ok
    except Exception:
        return False


def show_ollama_diagnostics(*, force_refresh: bool = False):
    """Ollama サーバーの現在状況をログに出力する。"""

    info = describe_server(force_refresh=force_refresh)
    log(
        "INFO",
        "Ollama診断: "
        f"host={info.get('host')} endpoint={info.get('endpoint')} checked_at={info.get('checked_at')}",
    )
    if info.get("reachable"):
        version = info.get("version") or "(バージョン情報なし)"
        models = info.get("models") or []
        if models:
            listed = ", ".join(models[:6])
            if len(models) > 6:
                listed += f" ... (+{len(models) - 6}件)"
        else:
            listed = "(登録モデルなし)"
        log("INFO", f"  version={version}")
        log("INFO", f"  models={listed}")
        if models and OLLAMA_MODEL not in models:
            log(
                "WARN",
                f"  選択中の OLLAMA_MODEL が現在のサーバーで確認できません。 `ollama run {OLLAMA_MODEL}` "
                "などで事前に pull されているか確認してください。",
            )
    else:
        log("ERR", "  Ollamaサーバーに接続できませんでした。--host や --endpoint を確認してください。")
    if info.get("version_error"):
        log("ERR", f"  version_error={info['version_error']}")
    if info.get("models_error"):
        log("ERR", f"  models_error={info['models_error']}")
    return info

def run_tts(text: str):
    from ai_talk.tts_voicevox import synthesize
    from ai_talk.audio_player import AudioPlayer
    setup(verbose=not args.quiet, color=not args.no_color)
    log("INFO", f"VOICEVOX={VOICEVOX_URL}")
    wav = synthesize(text or "テストです。音声合成。")
    ap = AudioPlayer(reporter=Reporter())
    ap.enqueue({"wav": wav, "text": text or "テスト"})
    ap._q.join()
    ap.stop(); ap.join(timeout=2)

def run_pipeline(initial_prompt: str):
    from ai_talk.pipeline import TalkPipeline
    setup(verbose=not args.quiet, color=not args.no_color)
    log(
        "INFO",
        f"VOICEVOX={VOICEVOX_URL}  OLLAMA={OLLAMA_HOST}  ENDPOINT={OLLAMA_GENERATE_PATH}  MODEL={OLLAMA_MODEL}",
    )
    show_ollama_diagnostics(force_refresh=args.inspect)
    rep = Reporter()
    tp = TalkPipeline(system_prompt="日本語で簡潔に答える。", reporter=rep)
    try:
        if initial_prompt:
            tp.push_user_text(initial_prompt)
            while True:
                time.sleep(0.2)
        else:
            log("INFO", "対話モード。'exit' で終了。")
            while True:
                s = input("> ").strip()
                if s.lower() in {"exit","quit"}: break
                if s: tp.push_user_text(s)
    except KeyboardInterrupt:
        pass
    finally:
        tp.close()

def run_asr(vosk_model_path: str):
    from ai_talk.main_demo import asr_demo
    setup(verbose=not args.quiet, color=not args.no_color)
    if not vosk_model_path:
        log("ERR", "--vosk-model を指定してください。")
        return
    asr_demo(vosk_model_path)

if __name__ == "__main__":
    if args.inspect and args.mode != "pipeline":
        # pipeline モードでは run_pipeline 内で詳細を表示するため、ここでは重複回避。
        setup(verbose=not args.quiet, color=not args.no_color)
        show_ollama_diagnostics(force_refresh=True)
    if args.mode=="tts":
        run_tts(args.text)
    elif args.mode=="pipeline":
        run_pipeline(args.prompt)
    else:
        run_asr(args.vosk_model)
