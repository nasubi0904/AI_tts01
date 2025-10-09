
"""
Ollama クライアント。ストリーミングを文単位で yield。
"""
import requests, json, re
from typing import Iterable
from copy import deepcopy
from .config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_PAYLOAD_OVERRIDES,
)


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
    payload["prompt"] = prompt
    payload["stream"] = stream
    if system:
        payload["system"] = system
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
    return payload

def generate(prompt: str, system: str = "") -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = _build_payload(prompt, system, stream=False)
    # タイムアウトは 120 秒。短すぎると長い応答で落ちる可能性があるため、
    # デフォルト値より長めに設定している。
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def stream(prompt: str, system: str = "") -> Iterable[str]:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = _build_payload(prompt, system, stream=True)
    # stream=True で逐次受信し、timeout=None にして推論終了まで接続を維持する。
    with requests.post(url, json=payload, stream=True, timeout=None) as r:
        r.raise_for_status()
        buf = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # まれに空行やJSON以外の文字列が混ざるため、例外は握りつぶして次へ。
                continue
            if "response" in obj:
                buf += obj["response"]
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
