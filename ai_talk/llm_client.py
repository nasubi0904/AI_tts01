"""Ollama クライアント。低遅延ストリーミングと診断機能を提供する。"""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from typing import Iterable, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

from .config import (
    OLLAMA_GENERATE_PATH,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_PAYLOAD_OVERRIDES,
)
from .logger import log


# ---- HTTP セッション共有と診断キャッシュ --------------------------------------------
_SESSION = Session()
_SESSION.headers.update({"Accept": "application/json"})
_SESSION.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
_SESSION.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
_SERVER_CACHE: dict[str, object] | None = None
_SENTENCE_BOUNDARY = re.compile(r"(.+?[。！？])")


def _session_post(url: str, payload: dict, *, stream: bool, timeout: float | None):
    """Session を介した POST リクエスト。"""

    return _SESSION.post(url, json=payload, stream=stream, timeout=timeout)


def _session_get(url: str, *, timeout: float):
    """診断情報取得用の GET リクエスト。"""

    return _SESSION.get(url, timeout=timeout)


def _is_chat_endpoint() -> bool:
    """OLLAMA_GENERATE_PATH が /api/chat 系かどうかを判定する。"""

    parsed = urlparse(OLLAMA_GENERATE_PATH)
    path = parsed.path or OLLAMA_GENERATE_PATH
    normalized = path.split("?", 1)[0].rstrip("/").lower()
    return normalized.endswith("/api/chat")


def _endpoint_candidates() -> List[Tuple[str, bool | None]]:
    """利用候補となるエンドポイント URL を列挙する。"""

    configured = _resolve_generate_url()
    candidates: List[Tuple[str, bool | None]] = [(configured, None)]
    if not _is_chat_endpoint():
        fallback = _resolve_host_path("/api/chat")
        if fallback != configured:
            candidates.append((fallback, True))
    return candidates


def _extract_error_message(response: requests.Response) -> str:
    """エラーレスポンス内のメッセージを抽出する。"""

    try:
        data = response.json()
    except ValueError:
        return response.text[:200]

    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return response.text[:200]


def _is_model_missing(response: requests.Response) -> Tuple[bool, str]:
    """404 がモデル未取得に起因するかどうかを判定する。"""

    if response.status_code != 404:
        return False, ""

    message = _extract_error_message(response)
    lowered = message.lower()
    keywords = ["model", "not found"]
    if all(word in lowered for word in keywords):
        return True, message
    return False, message


def _extract_response_text(data: dict) -> str:
    """generate/chat 双方のレスポンスからテキスト部分のみを取り出す。"""

    if not isinstance(data, dict):
        return ""
    if isinstance(data.get("response"), str):
        return data["response"]
    message = data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _build_payload(
    prompt: str,
    system: str,
    stream: bool,
    *,
    force_chat: bool | None = None,
) -> dict:
    """Ollama へ送る payload を組み立てる。"""

    use_chat_schema = _is_chat_endpoint() if force_chat is None else force_chat
    payload = {"model": OLLAMA_MODEL}
    payload_options = {}
    if OLLAMA_PAYLOAD_OVERRIDES:
        base = deepcopy(OLLAMA_PAYLOAD_OVERRIDES)
        if isinstance(base.get("options"), dict):
            payload_options = base.pop("options")
        payload.update(base)
    payload["stream"] = stream
    options = {}
    if isinstance(payload_options, dict):
        options.update(payload_options)
    if OLLAMA_OPTIONS:
        options.update(OLLAMA_OPTIONS)
    if options:
        payload["options"] = options
    if use_chat_schema:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = []
        if system and not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload["messages"] = messages
        payload.pop("prompt", None)
        payload.pop("system", None)
    else:
        payload["prompt"] = prompt
        if system:
            payload["system"] = system
        else:
            payload.pop("system", None)
    return payload


def _resolve_generate_url() -> str:
    if OLLAMA_GENERATE_PATH.startswith("http://") or OLLAMA_GENERATE_PATH.startswith("https://"):
        return OLLAMA_GENERATE_PATH
    return urljoin(OLLAMA_HOST + "/", OLLAMA_GENERATE_PATH.lstrip("/"))


def _resolve_host_path(path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return urljoin(OLLAMA_HOST + "/", path.lstrip("/"))


def _raise_with_hint(
    response: requests.Response,
    payload: dict,
    *,
    error_message: str | None = None,
) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        hint = [
            "Ollama サーバーへのリクエストに失敗しました。",
            f"URL={response.request.url}",
        ]
        if response.status_code == 404:
            if error_message:
                hint.append(f"サーバーからの応答: {error_message}")
            hint.append(
                "エンドポイントが存在しない、もしくは指定モデルが未ダウンロードの可能性があります。\n"
                "OLLAMA_HOST / OLLAMA_GENERATE_PATH / OLLAMA_MODEL を見直してください。",
            )
        elif response.status_code == 401:
            hint.append("認証が必要な環境の場合はトークン設定を確認してください。")
        else:
            hint.append(f"status={response.status_code} body={response.text[:200]}")
        hint.append(f"payload keys={sorted(payload.keys())}")
        raise requests.HTTPError("\n".join([str(exc)] + hint), response=response) from None


def describe_server(*, force_refresh: bool = False, timeout: float = 3.0) -> dict:
    """Ollama サーバーのバージョンやモデル一覧を収集する。"""

    global _SERVER_CACHE
    if not force_refresh and _SERVER_CACHE is not None:
        return _SERVER_CACHE

    info: dict[str, object] = {
        "host": OLLAMA_HOST,
        "endpoint": _resolve_generate_url(),
        "reachable": False,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "",
        "models": [],
    }

    version_url = _resolve_host_path("/api/version")
    try:
        r = _session_get(version_url, timeout=timeout)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("version"), str):
                info["version"] = data["version"]
            info["reachable"] = True
        else:
            info["version_error"] = f"status={r.status_code} body={r.text[:120]}"
    except RequestException as exc:
        info["version_error"] = str(exc)

    tags_url = _resolve_host_path("/api/tags")
    try:
        r = _session_get(tags_url, timeout=timeout)
        if r.ok:
            data = r.json()
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                names: list[str] = []
                for model in data["models"]:
                    if isinstance(model, dict):
                        name = model.get("name")
                        if isinstance(name, str):
                            names.append(name)
                if names:
                    info["models"] = names
                    info["reachable"] = True
        else:
            info["models_error"] = f"status={r.status_code}"
    except RequestException as exc:
        info["models_error"] = str(exc)

    _SERVER_CACHE = info
    return info


def generate(prompt: str, system: str = "") -> str:
    candidates = _endpoint_candidates()
    last_error: requests.HTTPError | None = None

    for idx, (url, force_chat) in enumerate(candidates):
        payload = _build_payload(prompt, system, stream=False, force_chat=force_chat)
        with _session_post(url, payload, stream=False, timeout=120) as response:
            missing_model, message = _is_model_missing(response)
            try:
                _raise_with_hint(response, payload, error_message=message if message else None)
            except requests.HTTPError as exc:
                last_error = exc
                if (
                    exc.response is not None
                    and exc.response.status_code == 404
                    and not missing_model
                    and idx + 1 < len(candidates)
                ):
                    next_url = candidates[idx + 1][0]
                    log(
                        "INFO",
                        "OLLAMA_GENERATE_PATH が 404 を返したため "
                        f"{next_url} へ自動切替して再試行します。",
                    )
                    continue
                if missing_model:
                    log(
                        "ERR",
                        "Ollama へ指定したモデルが存在しません。"
                        f" model={OLLAMA_MODEL} message={message}",
                    )
                raise
            data = response.json()
            return _extract_response_text(data)

    if last_error is not None:
        raise last_error
    return ""


def stream(prompt: str, system: str = "") -> Iterable[str]:
    candidates = _endpoint_candidates()
    last_error: requests.HTTPError | None = None

    for idx, (url, force_chat) in enumerate(candidates):
        payload = _build_payload(prompt, system, stream=True, force_chat=force_chat)
        response = _session_post(url, payload, stream=True, timeout=None)
        missing_model, message = _is_model_missing(response)
        try:
            _raise_with_hint(response, payload, error_message=message if message else None)
        except requests.HTTPError as exc:
            response.close()
            last_error = exc
            if (
                exc.response is not None
                and exc.response.status_code == 404
                and not missing_model
                and idx + 1 < len(candidates)
            ):
                next_url = candidates[idx + 1][0]
                log(
                    "INFO",
                    "OLLAMA_GENERATE_PATH が 404 を返したため "
                    f"{next_url} へ自動切替して再試行します。",
                )
                continue
            if missing_model:
                log(
                    "ERR",
                    "Ollama へ指定したモデルが存在しません。"
                    f" model={OLLAMA_MODEL} message={message}",
                )
            raise

        with response:
            buffer = ""
            for line in response.iter_lines(decode_unicode=True, chunk_size=128):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = _extract_response_text(obj)
                if chunk:
                    sentences, buffer = _collect_sentences(buffer + chunk)
                    for sentence in sentences:
                        yield sentence
                if obj.get("done"):
                    tail = buffer.strip()
                    if tail:
                        yield tail
                    break
            return

    if last_error is not None:
        raise last_error


def _collect_sentences(buffer: str) -> tuple[list[str], str]:
    sentences: list[str] = []
    last_end = 0
    for match in _SENTENCE_BOUNDARY.finditer(buffer):
        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)
        last_end = match.end()
    return sentences, buffer[last_end:]
