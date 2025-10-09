#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_ai_talk_test_v4.py
VSCodeからの単体テスト。CLI可視化強化版。
"""
import argparse, os, time

# ---- 先に引数を読み、環境変数を上書きしてから ai_talk を import ----
ap = argparse.ArgumentParser(description="ai_talk v4 単体テスト")
ap.add_argument("--mode", choices=["tts","pipeline","asr"], default="pipeline")
ap.add_argument("--text", default="")
ap.add_argument("--prompt", default="")
ap.add_argument("--vosk-model", default="")
ap.add_argument("--model", default="", help="Ollamaモデル名を一時上書き")
ap.add_argument("--host", default="", help="OllamaホストURLを一時上書き")
ap.add_argument("--options", default="", help="OLLAMA_OPTIONS_JSON を一時上書き (JSON文字列)")
ap.add_argument("--quiet", action="store_true", help="ログ最小化")
ap.add_argument("--no-color", action="store_true", help="ANSI色を無効化")
args = ap.parse_args()

if args.model:
    os.environ["OLLAMA_MODEL"] = args.model
if args.host:
    os.environ["OLLAMA_HOST"] = args.host
if args.options:
    os.environ["OLLAMA_OPTIONS_JSON"] = args.options
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
