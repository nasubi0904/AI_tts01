#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_ai_talk.py
接続・設定・最小経路の動作確認。モデル・ホストをCLIで上書き可能。

例:
  python diagnose_ai_talk.py --model gpt-oss:20b
  python diagnose_ai_talk.py --host http://127.0.0.1:11434 --model llama3.1:8b
"""
import os
import argparse
import requests

# ---- CLI引数 ----
ap = argparse.ArgumentParser(description="ai_talk 診断")
ap.add_argument("--model", default="", help="Ollamaで使用するモデル名。未指定なら環境変数/設定を使用")
ap.add_argument("--host", default="", help="OllamaホストURL。例 http://127.0.0.1:11434")
ap.add_argument("--prompt", default="5文字で日本語の挨拶を一つ。", help="generateテストのプロンプト")
args = ap.parse_args()

# ---- パッケージ設定読み込み ----
from ai_talk.config import VOICEVOX_URL, VOICEVOX_SPEAKER_ID, OLLAMA_HOST as CFG_HOST, OLLAMA_MODEL as CFG_MODEL
from ai_talk.tts_voicevox import synthesize
from ai_talk.audio_player import AudioPlayer

# 引数で上書き（未指定なら env→config の順）
OLLAMA_HOST = args.host or os.getenv("OLLAMA_HOST", CFG_HOST)
OLLAMA_MODEL = args.model or os.getenv("OLLAMA_MODEL", CFG_MODEL)

print("[CONFIG] VOICEVOX_URL:", VOICEVOX_URL)
print("[CONFIG] VOICEVOX_SPEAKER_ID:", VOICEVOX_SPEAKER_ID)
print("[USE   ] OLLAMA_HOST:", OLLAMA_HOST)
print("[USE   ] OLLAMA_MODEL:", OLLAMA_MODEL)

# 1) VOICEVOX /speakers
try:
    r = requests.get(f"{VOICEVOX_URL}/speakers", timeout=3)
    r.raise_for_status()
    speakers = r.json()
    all_ids = []
    for s in speakers:
        for st in s.get('styles', []):
            all_ids.append(st.get('id'))
    print(f"[OK ] VOICEVOX /speakers count={len(speakers)} example_ids={all_ids[:5]}")
    if VOICEVOX_SPEAKER_ID not in all_ids:
        print(f"[WARN] VOICEVOX_SPEAKER_ID={VOICEVOX_SPEAKER_ID} が見つからない可能性。別ID推奨。")
except Exception as e:
    print("[NG ] VOICEVOX /speakers:", e)

# 2) VOICEVOX synth + 再生
try:
    wav = synthesize("テストです。音声合成。")
    if wav and wav[:4] == b'RIFF':
        print("[OK ] VOICEVOX synthesis bytes:", len(wav))
        aply = AudioPlayer()
        aply.enqueue(wav)
        aply._q.join()
        aply.stop(); aply.join(timeout=2)
        print("[OK ] 再生完了")
    else:
        print("[NG ] synthesis からWAV未取得")
except Exception as e:
    print("[NG ] VOICEVOX synthesis:", e)

# 3) Ollama /api/tags
try:
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    r.raise_for_status()
    tags = r.json()
    names = [m.get('name') for m in tags.get('models', [])]
    print(f"[OK ] Ollama /api/tags models={names}")
    if OLLAMA_MODEL not in names:
        print(f"[WARN] 指定モデル {OLLAMA_MODEL} は未pullの可能性。'ollama pull {OLLAMA_MODEL}'")
except Exception as e:
    print("[NG ] Ollama /api/tags:", e)

# 4) Ollama /api/generate
try:
    payload = {"model": OLLAMA_MODEL, "prompt": args.prompt, "stream": False}
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response","")
    print("[OK ] generate len:", len(resp))
    print("      sample:", resp[:60].replace("\n"," "))
except Exception as e:
    print("[NG ] Ollama /api/generate:", e)
