"""Ollama との通信を抽象化した LLM クライアント実装。"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException

from .config import (
    OLLAMA_GENERATE_PATH,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_MODELFILE_OPTIONS,
    OLLAMA_OPTIONS,
    OLLAMA_PAYLOAD_OVERRIDES,
)
from .logger import log


# --------------------------------------------------------------------------------------
# 共通ユーティリティ


_SENTENCE_BOUNDARY = re.compile(r"(.+?[。！？])")


def _create_default_session() -> Session:
    session = Session()
    session.headers.update({"Accept": "application/json"})
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _extract_response_text(data: Mapping[str, object] | Sequence[object] | None) -> str:
    if not isinstance(data, Mapping):
        return ""
    response = data.get("response")
    if isinstance(response, str):
        return response
    message = data.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _collect_sentences(buffer: str) -> tuple[list[str], str]:
    sentences: list[str] = []
    last_end = 0
    for match in _SENTENCE_BOUNDARY.finditer(buffer):
        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)
        last_end = match.end()
    return sentences, buffer[last_end:]


# --------------------------------------------------------------------------------------
# 設定およびエンドポイント解決


@dataclass(frozen=True)
class EndpointCandidate:
    """利用候補となるエンドポイントとチャットスキーマ強制フラグ。"""

    url: str
    force_chat: bool | None


@dataclass(frozen=True)
class OllamaSettings:
    """Ollama への接続設定を保持するデータクラス。"""

    host: str
    generate_path: str
    model: str
    options: Mapping[str, object] | None = None
    modelfile_options: Mapping[str, object] | None = None
    payload_overrides: Mapping[str, object] | None = None

    @classmethod
    def from_env(cls) -> "OllamaSettings":
        return cls(
            host=OLLAMA_HOST,
            generate_path=OLLAMA_GENERATE_PATH,
            model=OLLAMA_MODEL,
            options=OLLAMA_OPTIONS or None,
            modelfile_options=OLLAMA_MODELFILE_OPTIONS or None,
            payload_overrides=OLLAMA_PAYLOAD_OVERRIDES or None,
        )

    # ------------------------------------------------------------------ URL 解決
    def resolve_host_path(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(self.host.rstrip("/") + "/", path.lstrip("/"))

    def resolve_generate_url(self) -> str:
        if self.generate_path.startswith("http://") or self.generate_path.startswith("https://"):
            return self.generate_path
        return self.resolve_host_path(self.generate_path)

    def is_chat_endpoint(self, *, path: str | None = None) -> bool:
        target = path if path is not None else self.generate_path
        parsed = urlparse(target)
        normalized = (parsed.path or target).split("?", 1)[0].rstrip("/").lower()
        return normalized.endswith("/api/chat")

    # ----------------------------------------------------------------- payload 構築
    def build_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        stream: bool,
        force_chat: bool | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"model": self.model, "stream": stream}

        overrides: dict[str, object] = {}
        if isinstance(self.payload_overrides, Mapping):
            overrides.update(self.payload_overrides)
        options_from_overrides = overrides.pop("options", None)
        payload.update(overrides)

        options: dict[str, object] = {}
        if isinstance(options_from_overrides, Mapping):
            options.update(options_from_overrides)
        if isinstance(self.options, Mapping):
            options.update(self.options)
        if isinstance(self.modelfile_options, Mapping):
            options.update(self.modelfile_options)
        if options:
            payload["options"] = options

        use_chat_schema = self.is_chat_endpoint() if force_chat is None else force_chat
        if use_chat_schema:
            payload_messages: list[dict[str, str]] = []
            existing = payload.get("messages")
            if isinstance(existing, list):
                payload_messages.extend(
                    {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
                    for msg in existing
                    if isinstance(msg, Mapping)
                )
            for message in messages:
                payload_messages.append(
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                )
            payload["messages"] = payload_messages
            payload.pop("prompt", None)
            payload.pop("system", None)
            return payload

        system_text = ""
        content_lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_text = content
            elif role == "user":
                content_lines.append(f"user: {content}")
            elif role == "assistant":
                content_lines.append(f"assistant: {content}")
            else:
                content_lines.append(content)

        payload["prompt"] = "\n".join(content_lines) + ("\nassistant:" if content_lines else "")
        if system_text:
            payload["system"] = system_text
        else:
            payload.pop("system", None)
        return payload

    # ---------------------------------------------------------------- 候補エンドポイント
    def endpoint_candidates(self) -> list[EndpointCandidate]:
        primary = self.resolve_generate_url()
        candidates: list[EndpointCandidate] = [EndpointCandidate(primary, None)]
        if not self.is_chat_endpoint(path=self.generate_path):
            fallback = self.resolve_host_path("/api/chat")
            if fallback != primary:
                candidates.append(EndpointCandidate(fallback, True))
        return candidates


# --------------------------------------------------------------------------------------
# HTTP クライアント層


@dataclass
class OllamaHTTPClient:
    """requests.Session をラップし、共通の HTTP 処理を提供する。"""

    session: Session = field(default_factory=_create_default_session)

    def post(self, url: str, payload: Mapping[str, object], *, stream: bool, timeout: float | None) -> Response:
        return self.session.post(url, json=payload, stream=stream, timeout=timeout)

    def get(self, url: str, *, timeout: float) -> Response:
        return self.session.get(url, timeout=timeout)


# --------------------------------------------------------------------------------------
# エラー解析ユーティリティ


def _extract_error_message(response: Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:200]

    if isinstance(data, Mapping):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return response.text[:200]


def _is_model_missing(response: Response) -> tuple[bool, str]:
    if response.status_code != 404:
        return False, ""
    message = _extract_error_message(response)
    lowered = message.lower()
    if "model" in lowered and "not" in lowered and "found" in lowered:
        return True, message
    return False, message


def _raise_with_hint(response: Response, payload: Mapping[str, object], *, error_message: str | None = None) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - requests が例外を投げる経路のみ
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


# --------------------------------------------------------------------------------------
# Ollama サービス本体


@dataclass
class OllamaService:
    """Ollama API へのアクセスとレスポンス処理を司るサービス。"""

    settings: OllamaSettings
    http: OllamaHTTPClient = field(default_factory=OllamaHTTPClient)
    _server_cache: dict[str, object] | None = field(default=None, init=False, repr=False)

    # ----------------------------------------------------------------- リクエスト共通
    def _perform_request(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        stream: bool,
        timeout: float | None,
    ) -> Iterator[str] | str:
        candidates = self.settings.endpoint_candidates()
        last_error: Exception | None = None

        for idx, candidate in enumerate(candidates):
            payload = self.settings.build_payload(messages, stream=stream, force_chat=candidate.force_chat)
            try:
                response = self.http.post(candidate.url, payload, stream=stream, timeout=timeout)
            except RequestException as exc:
                log(
                    "ERR",
                    "Ollama への接続に失敗しました。"
                    f" url={candidate.url} error={exc}",
                )
                last_error = exc
                continue
            missing_model, message = _is_model_missing(response)
            try:
                _raise_with_hint(response, payload, error_message=message or None)
            except requests.HTTPError as exc:
                response.close()
                last_error = exc
                if (
                    exc.response is not None
                    and exc.response.status_code == 404
                    and not missing_model
                    and idx + 1 < len(candidates)
                ):
                    next_url = candidates[idx + 1].url
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
                        f" model={self.settings.model} message={message}",
                    )
                raise

            if stream:
                return self._iter_stream(response)
            try:
                data = response.json()
            except ValueError:
                text = response.text
            else:
                text = _extract_response_text(data) or ""
                if not text:
                    text = response.text
            finally:
                response.close()
            return text

        if last_error is not None:
            raise last_error
        return iter(()) if stream else ""

    # ---------------------------------------------------------------- ストリーム処理
    def _iter_stream(self, response: Response) -> Iterator[str]:
        def _generator() -> Iterator[str]:
            with response:
                for line in response.iter_lines(decode_unicode=True, chunk_size=128):
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = _extract_response_text(payload)
                    if chunk:
                        yield chunk
                    if isinstance(payload, Mapping) and payload.get("done"):
                        break

        return _generator()

    # ----------------------------------------------------------------- 公開 API
    def request_text(self, messages: Sequence[Mapping[str, str]]) -> str:
        result = self._perform_request(messages, stream=False, timeout=120)
        return result if isinstance(result, str) else ""

    def request_stream(self, messages: Sequence[Mapping[str, str]]) -> Iterator[str]:
        result = self._perform_request(messages, stream=True, timeout=None)
        return result if isinstance(result, Iterator) else iter(())

    # ----------------------------------------------------------------- 診断情報
    def describe_server(self, *, force_refresh: bool = False, timeout: float = 3.0) -> dict[str, object]:
        if not force_refresh and self._server_cache is not None:
            return self._server_cache

        info: dict[str, object] = {
            "host": self.settings.host,
            "endpoint": self.settings.resolve_generate_url(),
            "reachable": False,
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "",
            "models": [],
        }

        version_url = self.settings.resolve_host_path("/api/version")
        try:
            response = self.http.get(version_url, timeout=timeout)
            if response.ok:
                data = response.json()
                if isinstance(data, Mapping) and isinstance(data.get("version"), str):
                    info["version"] = data["version"]
                info["reachable"] = True
            else:
                info["version_error"] = f"status={response.status_code} body={response.text[:120]}"
        except RequestException as exc:
            info["version_error"] = str(exc)

        tags_url = self.settings.resolve_host_path("/api/tags")
        try:
            response = self.http.get(tags_url, timeout=timeout)
            if response.ok:
                data = response.json()
                if isinstance(data, Mapping) and isinstance(data.get("models"), list):
                    names: list[str] = []
                    for item in data["models"]:
                        if isinstance(item, Mapping):
                            name = item.get("name")
                            if isinstance(name, str):
                                names.append(name)
                    if names:
                        info["models"] = names
                        info["reachable"] = True
            else:
                info["models_error"] = f"status={response.status_code}"
        except RequestException as exc:
            info["models_error"] = str(exc)

        self._server_cache = info
        return info


# --------------------------------------------------------------------------------------
# グローバルエントリポイント


_GLOBAL_SERVICE = OllamaService(OllamaSettings.from_env())


def describe_server(*, force_refresh: bool = False, timeout: float = 3.0) -> dict:
    return _GLOBAL_SERVICE.describe_server(force_refresh=force_refresh, timeout=timeout)


# --------------------------------------------------------------------------------------
# チャットセッションユーティリティ


@dataclass
class OllamaChatSession:
    """会話履歴を保持しつつ OllamaService を利用するチャットセッション。"""

    system_prompt: str = ""
    service: OllamaService = field(default_factory=lambda: _GLOBAL_SERVICE)
    messages: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        system = self.system_prompt.strip()
        if system:
            self.messages.append({"role": "system", "content": system})

    # ---------------------------------------------------------------- util
    def reset(self) -> None:
        system = self.system_prompt.strip()
        self.messages.clear()
        if system:
            self.messages.append({"role": "system", "content": system})

    def _append_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def _append_assistant(self, text: str) -> None:
        if text.strip():
            self.messages.append({"role": "assistant", "content": text})

    # ---------------------------------------------------------------- public
    def stream_sentences(self, user_text: str) -> Iterable[str]:
        clean = user_text.strip()
        if not clean:
            return

        self._append_user(clean)

        buffer = ""
        collected: list[str] = []
        try:
            for chunk in self.service.request_stream(self.messages):
                if not chunk:
                    continue
                collected.append(chunk)
                sentences, buffer = _collect_sentences(buffer + chunk)
                for sentence in sentences:
                    yield sentence
            tail = buffer.strip()
            if tail:
                collected.append(tail)
                yield tail
        except Exception:
            if self.messages:
                self.messages.pop()
            raise

        assistant_text = "".join(collected)
        self._append_assistant(assistant_text)


__all__ = [
    "OllamaChatSession",
    "OllamaHTTPClient",
    "OllamaService",
    "OllamaSettings",
    "describe_server",
]

