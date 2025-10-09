
"""
Ollama クライアント。ストリーミングを文単位で yield。

エンドポイントの切替や診断情報の取得など、Ollama サーバーに関する
補助的な機能もここで一元管理する。
"""
import requests, json, re, time
from typing import Iterable, List, Tuple
from copy import deepcopy
from urllib.parse import urljoin, urlparse
from requests import Session
from requests.exceptions import RequestException
from .config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_PAYLOAD_OVERRIDES,
    OLLAMA_GENERATE_PATH,
)
from .logger import log


# ---- HTTP セッションの共有と診断キャッシュ -------------------------------------------
#
# 毎回 requests.post を呼び出すと TCP コネクションが張り直され、推論待ちのたびに
# 遅延が増えることがある。Session を共有することで Keep-Alive を活用し、応答の
# 安定化と効率化を図る。同時にサーバー診断情報のキャッシュも管理する。
_SESSION = Session()
_SERVER_CACHE: dict[str, object] | None = None


def _session_post(url: str, payload: dict, *, stream: bool, timeout: float):
    """Session を介した POST リクエスト。"""

    return _SESSION.post(url, json=payload, stream=stream, timeout=timeout)


def _session_get(url: str, *, timeout: float):
    """診断情報取得用の GET リクエスト。"""

    return _SESSION.get(url, timeout=timeout)



def _is_chat_endpoint() -> bool:
    """OLLAMA_GENERATE_PATH が /api/chat 系かどうかを判定する。"""

    parsed = urlparse(OLLAMA_GENERATE_PATH)
    # 相対パスの場合は path に格納される。絶対URLなら path 部分のみを見る。
    path = parsed.path or OLLAMA_GENERATE_PATH
    normalized = path.split("?", 1)[0].rstrip("/").lower()
    return normalized.endswith("/api/chat")


def _endpoint_candidates() -> List[Tuple[str, bool | None]]:
    """利用候補となるエンドポイントURLを列挙する。"""

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
    # Ollama 公式実装では、存在しないモデルを指定した際に
    # "model 'xxx' not found" と返すため、それらの文言を検知する。
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


def _build_payload(prompt: str, system: str, stream: bool, *, force_chat: bool | None = None) -> dict:
    """Ollamaへ送るpayloadを組み立てる。

    payloadに含められる主な項目は以下の通り。
    - model: 使用するモデル名。環境変数 OLLAMA_MODEL で管理。
    - prompt: ユーザー入力。文字列。
    - system: システムプロンプト。空文字列の場合は送らない。
    - stream: ストリーミング応答を希望するかどうかの真偽値。
    - options: パラメータ辞書。num_predict, temperature などを指定可能。
    - そのほか keep_alive や format など、Ollama公式ドキュメントで定義される
      任意のキーを OLLAMA_PAYLOAD_JSON 側で追加できる。

    OLLAMA_PAYLOAD_JSON に {"options": {...}} を含めた場合は options の初期値に
    取り込み、最後に OLLAMA_OPTIONS_JSON の値で上書きする。これにより VSCode で
    編集したローカル設定 > コマンドライン引数 > 環境変数の順に優先度を揃えている。
    """

    use_chat_schema = _is_chat_endpoint() if force_chat is None else force_chat
    payload = {"model": OLLAMA_MODEL}
    payload_options = {}
    if OLLAMA_PAYLOAD_OVERRIDES:
        # deepcopy してから操作することで、設定を複数回利用しても副作用が出ないようにする。
        # 公式仕様にある top-level キー (model/prompt/system/options など) 以外もそのまま残す。
        # 例: {"keep_alive":"5m","format":"json"} を設定すればAPIリクエストに反映される。
        base = deepcopy(OLLAMA_PAYLOAD_OVERRIDES)
        if isinstance(base.get("options"), dict):
            # payload 側で options を内包する場合はここで退避させ、後段でマージする。
            payload_options = base.pop("options")
        payload.update(base)
    payload["stream"] = stream
    options = {}
    if isinstance(payload_options, dict):
        # payload 側で宣言された options をベースにする。VSCodeテンプレートや環境変数からの上書きを許可。
        options.update(payload_options)
    if OLLAMA_OPTIONS:
        # OLLAMA_OPTIONS_JSON (もしくは --options 引数) を最終上書きとして適用する。
        # 例えば CLI で {"num_predict":64} を渡せば、payload テンプレート側の値より優先される。
        options.update(OLLAMA_OPTIONS)
    if options:
        payload["options"] = options
    if use_chat_schema:
        # /api/chat を利用する場合は messages フィールドを組み立てる。
        # 既に payload 側で messages が指定されていれば尊重し、末尾にユーザー入力を追加する。
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = []
        # system プロンプトは先頭に1件だけ挿入する。既に system ロールがあれば重複を避ける。
        if system and not any(m.get("role") == "system" for m in messages if isinstance(m, dict)):
            messages.insert(0, {"role": "system", "content": system})
        # ユーザー入力は常に末尾に追加する。既存の履歴があっても最新発話として扱える。
        messages.append({"role": "user", "content": prompt})
        payload["messages"] = messages
        # chat schema では prompt/system はトップレベルで利用しないため削除して公式仕様に合わせる。
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
    """生成APIのURLを決定する。

    OLLAMA_GENERATE_PATH が絶対URLの場合はそれを優先し、相対パスの場合は
    OLLAMA_HOST との結合結果を返す。末尾のスラッシュは除去して冗長なリクエスト
    を避ける。
    """

    if OLLAMA_GENERATE_PATH.startswith("http://") or OLLAMA_GENERATE_PATH.startswith("https://"):
        return OLLAMA_GENERATE_PATH
    # urljoin は第二引数が '/' で始まるとルートへ張り付ける挙動を利用する。
    return urljoin(OLLAMA_HOST + "/", OLLAMA_GENERATE_PATH.lstrip("/"))


def _resolve_host_path(path: str) -> str:
    """ホストURLと任意のパスを結合するユーティリティ。"""

    if path.startswith("http://") or path.startswith("https://"):
        return path
    return urljoin(OLLAMA_HOST + "/", path.lstrip("/"))


def _raise_with_hint(
    response: requests.Response,
    payload: dict,
    *,
    error_message: str | None = None,
):
    """HTTPエラー時に接続設定の見直しポイントを添えて例外を投げ直す。"""

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
                "OLLAMA_HOST / OLLAMA_GENERATE_PATH / OLLAMA_MODEL を見直してください。"
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
                        "Ollama へ指定したモデルが存在しません。"\
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
                    "Ollama へ指定したモデルが存在しません。"\
                    f" model={OLLAMA_MODEL} message={message}",
                )
            raise

        with response:
            buf = ""
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # まれに空行やJSON以外の文字列が混ざるため、例外は握りつぶして次へ。
                    continue
                chunk = _extract_response_text(obj)
                if chunk:
                    buf += chunk
                    parts = re.split(r"([。！？])", buf)
                    for i in range(0, len(parts) - 1, 2):
                        sent = (parts[i] + parts[i + 1]).strip()
                        if sent:
                            yield sent
                    buf = "" if len(parts) % 2 == 0 else parts[-1]
                if obj.get("done"):
                    if buf.strip():
                        yield buf.strip()
                    break
            return

    if last_error is not None:
        raise last_error
