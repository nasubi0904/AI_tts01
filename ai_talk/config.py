
"""
共通設定。環境変数 .env / tts.env を読み込む。
"""
import os, json
from dotenv import load_dotenv

if os.path.exists("tts.env"):
    load_dotenv("tts.env")
if os.path.exists(".env"):
    load_dotenv(".env")

def _normalize_host(url: str) -> str:
    if not url:
        return "http://127.0.0.1:11434"
    u = url.strip().rstrip("/")
    if u.startswith("0.0.0.0:"):
        u = "127.0.0.1:" + u.split(":", 1)[1]
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "http://" + u

VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021").rstrip("/")
VOICEVOX_SPEAKER_ID = int(os.getenv("VOICEVOX_SPEAKER_ID", "1"))

OLLAMA_HOST = _normalize_host(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# 速度調整に使う Ollama options をJSON文字列で渡せる
# 例: OLLAMA_OPTIONS_JSON={"num_predict":128,"temperature":0.6}
try:
    OLLAMA_OPTIONS = json.loads(os.getenv("OLLAMA_OPTIONS_JSON","{}"))
except Exception:
    OLLAMA_OPTIONS = {}

ASR_DEVICE = os.getenv("ASR_DEVICE", "")
ASR_BLOCK_SIZE = int(os.getenv("ASR_BLOCK_SIZE", "1600"))
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
