
"""
Ollama クライアント。ストリーミングを文単位で yield。
"""
import requests, json, re
from typing import Iterable
from copy import deepcopy
from urllib.parse import urljoin, urlparse
from .config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_PAYLOAD_OVERRIDES,
    OLLAMA_GENERATE_PATH,
)


def _is_chat_endpoint() -> bool:
    """OLLAMA_GENERATE_PATH が /api/chat 系かどうかを判定する。"""

    parsed = urlparse(OLLAMA_GENERATE_PATH)
    # 相対パスの場合は path に格納される。絶対URLなら path 部分のみを見る。
    path = parsed.path or OLLAMA_GENERATE_PATH
    normalized = path.split("?", 1)[0].rstrip("/").lower()
    return normalized.endswith("/api/chat")


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


def _build_payload(prompt: str, system: str, stream: bool) -> dict:
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

    use_chat_schema = _is_chat_endpoint()
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


def _raise_with_hint(response: requests.Response, payload: dict):
    """HTTPエラー時に接続設定の見直しポイントを添えて例外を投げ直す。"""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        hint = [
            "Ollama サーバーへのリクエストに失敗しました。",
            f"URL={response.request.url}",
        ]
        if response.status_code == 404:
            hint.append(
                "エンドポイントが存在しない可能性があります。\n"
                "OLLAMA_HOST や OLLAMA_GENERATE_PATH を見直してください。"
            )
        elif response.status_code == 401:
            hint.append("認証が必要な環境の場合はトークン設定を確認してください。")
        else:
            hint.append(f"status={response.status_code} body={response.text[:200]}")
        hint.append(f"payload keys={sorted(payload.keys())}")
        raise requests.HTTPError("\n".join([str(exc)] + hint), response=response) from None


def generate(prompt: str, system: str = "") -> str:
    url = _resolve_generate_url()
    payload = _build_payload(prompt, system, stream=False)
    # タイムアウトは 120 秒。短すぎると長い応答で落ちる可能性があるため、
    # デフォルト値より長めに設定している。
    r = requests.post(url, json=payload, timeout=120)
    _raise_with_hint(r, payload)
    data = r.json()
    return _extract_response_text(data)

def stream(prompt: str, system: str = "") -> Iterable[str]:
    url = _resolve_generate_url()
    payload = _build_payload(prompt, system, stream=True)
    # stream=True で逐次受信し、timeout=None にして推論終了まで接続を維持する。
    with requests.post(url, json=payload, stream=True, timeout=None) as r:
        _raise_with_hint(r, payload)
        buf = ""
        for line in r.iter_lines(decode_unicode=True):
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
                for i in range(0, len(parts)-1, 2):
                    sent = (parts[i] + parts[i+1]).strip()
                    if sent:
                        yield sent
                buf = "" if len(parts) % 2 == 0 else parts[-1]
            if obj.get("done"):
                if buf.strip():
                    yield buf.strip()
                break
