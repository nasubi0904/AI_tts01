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
ap.add_argument("--quiet", action="store_true", help="ログ最小化")
ap.add_argument("--no-color", action="store_true", help="ANSI色を無効化")
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
#   1. コマンドライン引数 (--options / --payload)
#   2. ここで定義した LOCAL_OLLAMA_*
#   3. .env / tts.env に記載した環境変数
#   すべてJSONとして解釈され、辞書でなければ自動的に空辞書に戻るため安全に試行錯誤できる。
#
# ※ JSON 文字列化の際には ensure_ascii=False を指定しているため、日本語コメントを含めても文字化けしない。
LOCAL_OLLAMA_OPTIONS = {}
LOCAL_OLLAMA_PAYLOAD = {}

if LOCAL_OLLAMA_OPTIONS and not args.options:
    # 環境変数を直接上書きすることで ai_talk.config の初期化ロジックに乗せる。
    os.environ["OLLAMA_OPTIONS_JSON"] = json.dumps(LOCAL_OLLAMA_OPTIONS, ensure_ascii=False)
if LOCAL_OLLAMA_PAYLOAD and not args.payload:
    # payload 側も同様。options キーを含む場合は llm_client._build_payload がマージする。
    os.environ["OLLAMA_PAYLOAD_JSON"] = json.dumps(LOCAL_OLLAMA_PAYLOAD, ensure_ascii=False)

if args.model:
    # 一時的に別モデルを試したい時のための上書き。--options 等よりも優先度が低い。
    os.environ["OLLAMA_MODEL"] = args.model
if args.host:
    os.environ["OLLAMA_HOST"] = args.host
if args.options:
    # CLIから渡したJSON文字列をそのまま保存。エラー時は config 側で空辞書化される。
    os.environ["OLLAMA_OPTIONS_JSON"] = args.options
if args.payload:
    # payload は official schema をベースに自由に拡張できる。
    os.environ["OLLAMA_PAYLOAD_JSON"] = args.payload
if args.no_color:
    os.environ["NO_COLOR"] = "1"

from ai_talk.config import VOICEVOX_URL, OLLAMA_HOST, OLLAMA_MODEL
from ai_talk.logger import setup, log, Reporter

def _ping(url, path):
    import requests
    try:
        r = requests.get(f"{url}{path}", timeout=3)
        return r.ok
    except Exception:
        return False

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
    log("INFO", f"VOICEVOX={VOICEVOX_URL}  OLLAMA={OLLAMA_HOST}  MODEL={OLLAMA_MODEL}")
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
    if args.mode=="tts":
        run_tts(args.text)
    elif args.mode=="pipeline":
        run_pipeline(args.prompt)
    else:
        run_asr(args.vosk_model)
