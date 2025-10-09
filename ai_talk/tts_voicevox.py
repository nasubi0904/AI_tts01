
"""
VOICEVOX クライアント。WAVバイトを返す。
"""
import json, requests
from .config import VOICEVOX_URL, VOICEVOX_SPEAKER_ID

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})

def _init():
    try:
        _session.post(f"{VOICEVOX_URL}/initialize_speaker", params={"speaker": VOICEVOX_SPEAKER_ID}, timeout=3)
    except Exception:
        pass
_init()

def synthesize(text: str) -> bytes:
    if not text:
        return b""
    q = _session.post(f"{VOICEVOX_URL}/audio_query", params={"text": text, "speaker": VOICEVOX_SPEAKER_ID}, timeout=10)
    q.raise_for_status()
    query = q.json()
    query.update({
        "speedScale":       1.05,
        "intonationScale":  1.0,
        "prePhonemeLength": 0.08,
        "postPhonemeLength":0.08
    })
    s = _session.post(f"{VOICEVOX_URL}/synthesis", params={"speaker": VOICEVOX_SPEAKER_ID}, data=json.dumps(query), timeout=20)
    s.raise_for_status()
    return s.content
