
"""
共通設定。環境変数 .env / tts.env を読み込む。
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv

# Windows などでスクリプトを親ディレクトリから呼び出すと、カレントディレクトリに
# tts.env / .env が存在しないため読み込みに失敗する。そこで `config.py` 自身の配置
# （= ai_talk/ 配下）からリポジトリルートを逆算し、優先的にそちらを探す。
_CONFIG_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CONFIG_DIR.parent

def _load_dotenv_safe(candidate: Path):
    """指定パスにファイルがあれば読み込む。存在しないときは無視。"""

    if candidate.exists():
        # load_dotenv は既に設定された環境変数を上書きしないため、run_ai_talk_test_v4.py
        # などで事前に setenv 済みの値は維持される。複数候補を順番に読み込んでも安全。
        load_dotenv(candidate)

# 1. リポジトリルート（config.py からの相対）
_load_dotenv_safe(_ROOT_DIR / "tts.env")
_load_dotenv_safe(_ROOT_DIR / ".env")

# 2. カレントディレクトリ直下（従来の挙動を維持）
_load_dotenv_safe(Path("tts.env"))
_load_dotenv_safe(Path(".env"))

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

def _normalize_api_path(path: str, default: str = "/api/generate") -> str:
    """Ollama の API パスを正規化する。

    - 絶対URL (http://...) が指定された場合はそのまま利用する。
    - 相対パスの場合は必ず先頭を '/' に揃え、末尾のスラッシュを除去する。
    - 空文字列の場合は default (既定は /api/generate) を返す。
    """

    if not path:
        return default
    trimmed = path.strip()
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed.rstrip("/")
    if not trimmed.startswith("/"):
        trimmed = "/" + trimmed
    return trimmed.rstrip("/") or default

# 404 対策として generate 用エンドポイントを環境変数で切り替え可能にする。
# OLLAMA_GENERATE_PATH="/api/chat" のように指定すると llm_client 側で chat payload に自動変換される。
OLLAMA_GENERATE_PATH = _normalize_api_path(os.getenv("OLLAMA_GENERATE_PATH", "/api/generate"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# 速度調整に使う Ollama options をJSON文字列で渡せる
# 例: OLLAMA_OPTIONS_JSON={"num_predict":128,"temperature":0.6}
# エンドポイントを /api/chat などに切り替えたい場合は OLLAMA_GENERATE_PATH を設定する。

def _load_json_env(key: str) -> dict:
    """環境変数に格納されたJSON文字列を辞書として読み出す。

    不正なJSONや辞書以外の構造が入っていた場合でも例外を発生させず、
    空の辞書を返して後続処理を止めない。実運用時に環境変数の入力ミスが
    発生してもアプリ全体が停止しないよう、ここで安全側に倒す。
    """
    raw = os.getenv(key, "{}")
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass
    return {}

OLLAMA_OPTIONS = _load_json_env("OLLAMA_OPTIONS_JSON")
OLLAMA_PAYLOAD_OVERRIDES = _load_json_env("OLLAMA_PAYLOAD_JSON")

def _load_int_env(key: str, default: int = 0) -> int:
    """整数の環境変数を安全に読み取る。"""

    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value >= 0 else default
    except ValueError:
        return default


def _load_float_env(key: str, default: float = 0.0) -> float:
    """浮動小数の環境変数を安全に読み取る。"""

    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _timeout_value(seconds: float) -> float | None:
    """0以下の値は None (無制限) として扱うユーティリティ。"""

    return seconds if seconds and seconds > 0 else None


# LLM 応答を制御するための閾値。num_predict を極端に下げずとも応答を短くできるよう、
# 文数・文字数・時間の上限を環境変数から設定できるようにする。
OLLAMA_STREAM_SENTENCE_LIMIT = _load_int_env("OLLAMA_STREAM_SENTENCE_LIMIT", 0)
OLLAMA_STREAM_CHAR_LIMIT = _load_int_env("OLLAMA_STREAM_CHAR_LIMIT", 0)
OLLAMA_STREAM_TIMEOUT = _load_float_env("OLLAMA_STREAM_TIMEOUT", 0.0)
# リクエスト全体のタイムアウト秒数。0 以下なら requests 側の既定値 (無制限) を利用する。
_REQUEST_TIMEOUT_SEC = _timeout_value(_load_float_env("OLLAMA_REQUEST_TIMEOUT", 120.0))
OLLAMA_REQUEST_TIMEOUT = _REQUEST_TIMEOUT_SEC

ASR_DEVICE = os.getenv("ASR_DEVICE", "")
ASR_BLOCK_SIZE = int(os.getenv("ASR_BLOCK_SIZE", "1600"))
ASR_SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
