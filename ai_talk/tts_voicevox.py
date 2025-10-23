
"""VOICEVOX HTTP クライアント。"""

from __future__ import annotations

import json
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from .config import VOICEVOX_SPEAKER_ID, VOICEVOX_URL


_SESSION = requests.Session()
_SESSION.headers.update({"Content-Type": "application/json"})
_SESSION.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
_SESSION.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))


def _initialize() -> None:
    try:
        _SESSION.post(
            f"{VOICEVOX_URL}/initialize_speaker",
            params={"speaker": VOICEVOX_SPEAKER_ID},
            timeout=3,
        )
    except Exception:  # noqa: BLE001  起動直後の初期化失敗は許容
        pass


_initialize()


def synthesize(text: str) -> bytes:
    if not text:
        return b""
    query = _request_json(
        "audio_query",
        params={"text": text, "speaker": VOICEVOX_SPEAKER_ID},
        timeout=10,
    )
    query.update(
        {
            "speedScale": 1.05,
            "intonationScale": 1.0,
            "prePhonemeLength": 0.08,
            "postPhonemeLength": 0.08,
        }
    )
    response = _SESSION.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": VOICEVOX_SPEAKER_ID},
        data=json.dumps(query),
        timeout=20,
    )
    response.raise_for_status()
    return response.content


def _request_json(endpoint: str, *, params: dict[str, Any], timeout: float) -> dict:
    response = _SESSION.post(f"{VOICEVOX_URL}/{endpoint}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()
