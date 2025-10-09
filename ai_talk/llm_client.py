
"""
Ollama クライアント。ストリーミングを文単位で yield。
"""
import requests, json, re
from typing import Iterable
from .config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_OPTIONS

def generate(prompt: str, system: str = "") -> str:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if OLLAMA_OPTIONS:
        payload["options"] = OLLAMA_OPTIONS
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def stream(prompt: str, system: str = "") -> Iterable[str]:
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
    if system:
        payload["system"] = system
    if OLLAMA_OPTIONS:
        payload["options"] = OLLAMA_OPTIONS
    with requests.post(url, json=payload, stream=True, timeout=None) as r:
        r.raise_for_status()
        buf = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
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
