#!/usr/bin/env python3
"""Ollama の `ollama run` に寄せた軽量チャット CLI。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_KEEP_ALIVE, DEFAULT_TEMPERATURE, DEFAULT_TOP_P = "30m", 0.0, 1.0
REQUEST_TIMEOUT = 300.0


@dataclass
class CliConfig:
    """CLI で利用する各種設定値。"""

    model: str
    prompt: Optional[str]
    host: str
    system_prompt: Optional[str]
    seed: Optional[int]
    temperature: float
    top_p: float
    num_predict: Optional[int]
    keep_alive: str
    stream: bool
    as_json: bool
    show_payload: bool
    use_generate: bool


class OrunError(RuntimeError):
    """CLI 全体で共有する例外。"""


def build_chat_payload(
    model: str, messages: List[Dict[str, str]], options: Dict[str, object], keep_alive: str, stream: bool
) -> Dict[str, object]:
    """/api/chat へ送る JSON を整形する。"""

    return {
        "model": model,
        "messages": messages,
        "options": options,
        "keep_alive": keep_alive,
        "stream": stream,
    }


def build_generate_payload(
    model: str, prompt: str, options: Dict[str, object], keep_alive: str, stream: bool
) -> Dict[str, object]:
    """/api/generate へ送る JSON を整形する。"""

    return {
        "model": model,
        "prompt": prompt,
        "options": options,
        "keep_alive": keep_alive,
        "stream": stream,
    }


def clean_options(
    seed: Optional[int], temperature: float, top_p: float, num_predict: Optional[int]
) -> Dict[str, object]:
    """options フィールドに含める値をまとめる。"""

    opts: Dict[str, object] = {"temperature": temperature, "top_p": top_p}
    if seed is not None:
        opts["seed"] = seed
    if num_predict is not None:
        opts["num_predict"] = num_predict
    return opts


def post_json(url: str, payload: Dict[str, object], stream: bool) -> requests.Response:
    """POST リクエストを送信する。"""

    try:
        return requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=stream,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:  # pragma: no cover - network
        raise OrunError("Ollama サーバーに接続できません。ホストや起動状況を確認してください。") from exc
    except requests.exceptions.Timeout as exc:  # pragma: no cover - network
        raise OrunError("リクエストがタイムアウトしました。サーバー負荷やネットワークを確認してください。") from exc


def ensure_success(response: requests.Response) -> None:
    """HTTP ステータスを検証する。"""

    if response.status_code == 404:
        raise OrunError("404 Not Found: エンドポイントまたはモデル名を確認してください。")
    if response.status_code == 405:
        raise OrunError("405 Method Not Allowed: POST メソッドで送信しているか確認してください。")
    if response.status_code >= 400:
        try:
            detail = response.json().get("error")  # type: ignore[arg-type]
        except ValueError:
            detail = response.text
        raise OrunError(f"HTTP {response.status_code}: {detail}")


def extract_text(chunk: Dict[str, object]) -> str:
    """応答チャンクからテキストを抽出する。"""

    message = chunk.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    response = chunk.get("response")
    if isinstance(response, str):
        return response
    return ""


def parse_stream(lines: Iterable[str], as_json: bool) -> str:
    """ストリーミング応答を逐次出力する。"""

    collected: List[str] = []
    for raw in lines:
        if not raw:
            continue
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            print(raw, file=sys.stderr)
            continue
        if chunk.get("error"):
            raise OrunError(str(chunk["error"]))
        piece = extract_text(chunk)
        if as_json:
            print(json.dumps(chunk, ensure_ascii=False), flush=True)
        elif piece:
            print(piece, end="", flush=True)
        if piece:
            collected.append(piece)
        if chunk.get("done"):
            break
    if not as_json:
        print(flush=True)
    return "".join(collected)


def call_api(config: CliConfig, endpoint: str, payload: Dict[str, object]) -> str:
    """共通のリクエスト処理を行い最終テキストを返す。"""

    if config.show_payload:
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    response = post_json(endpoint, payload, stream=config.stream)
    ensure_success(response)
    if config.stream:
        return parse_stream(response.iter_lines(decode_unicode=True), config.as_json)
    try:
        body = response.json()
    except ValueError as exc:
        raise OrunError("サーバーから JSON 以外の応答を受信しました。") from exc
    if not isinstance(body, dict):
        raise OrunError("JSON オブジェクト以外の応答を受信しました。")
    if body.get("error"):
        raise OrunError(str(body["error"]))
    if config.as_json:
        print(json.dumps(body, ensure_ascii=False))
    text = extract_text(body)
    if text:
        if not config.as_json:
            print(text)
        return text
    fallback = json.dumps(body, ensure_ascii=False)
    if not config.as_json:
        print(fallback)
    return fallback


def dispatch_chat(config: CliConfig, messages: List[Dict[str, str]]) -> str:
    """チャット API を呼び出す。"""

    return call_api(
        config,
        f"{config.host}/api/chat",
        build_chat_payload(
            config.model,
            messages,
            clean_options(config.seed, config.temperature, config.top_p, config.num_predict),
            config.keep_alive,
            config.stream,
        ),
    )


def dispatch_generate(config: CliConfig, prompt: str) -> str:
    """生成 API を呼び出す。"""

    return call_api(
        config,
        f"{config.host}/api/generate",
        build_generate_payload(
            config.model,
            prompt,
            clean_options(config.seed, config.temperature, config.top_p, config.num_predict),
            config.keep_alive,
            config.stream,
        ),
    )


def run_repl(config: CliConfig) -> None:
    """REPL モードを処理する。"""

    history: List[Dict[str, str]] = []
    system_entry = {"role": "system", "content": config.system_prompt} if config.system_prompt else None
    if system_entry:
        history.append(system_entry)
    print("対話モードを開始します。/exit で終了、/reset で履歴を消去します。", file=sys.stderr)
    while True:
        try:
            user_input = input("> ")
        except EOFError:
            print(file=sys.stderr)
            break
        except KeyboardInterrupt:
            print(file=sys.stderr)
            break
        if not user_input:
            continue
        if user_input.strip() == "/exit":
            break
        if user_input.strip() == "/reset":
            history = []
            if system_entry:
                history.append(system_entry)
            print("会話履歴をリセットしました。", file=sys.stderr)
            continue
        user_message = {"role": "user", "content": user_input}
        reply = dispatch_chat(config, history + [user_message])
        history.append(user_message)
        if reply:
            history.append({"role": "assistant", "content": reply})


def run_once(config: CliConfig) -> None:
    """ワンショット要求を処理する。"""

    if config.prompt is None:
        raise OrunError("ワンショット実行にはプロンプトを指定してください。")
    if config.use_generate:
        dispatch_generate(config, config.prompt)
        return
    messages: List[Dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": config.prompt})
    dispatch_chat(config, messages)


def parse_args(argv: Optional[List[str]] = None) -> CliConfig:
    """CLI 引数を解析する。"""

    parser = argparse.ArgumentParser(description="Ollama 互換の簡易 CLI")
    parser.add_argument("model", help="使用するモデル名")
    parser.add_argument("prompt", nargs="?", help="ワンショットプロンプト")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama ホスト URL")
    parser.add_argument("--system", dest="system_prompt", help="system プロンプト")
    parser.add_argument("--seed", type=int, help="乱数シード")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="温度 (既定 0)")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="top-p (既定 1.0)")
    parser.add_argument("--num-predict", type=int, help="生成トークン数の上限")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE, help="keep-alive (既定 30m)")
    parser.add_argument("--no-stream", action="store_true", help="ストリーミングを無効化する")
    parser.add_argument("--json", action="store_true", help="応答を JSON で出力する")
    parser.add_argument("--show-payload", action="store_true", help="送信 JSON を表示する")
    parser.add_argument("--generate", action="store_true", help="/api/generate を使用する")
    args = parser.parse_args(argv)
    if args.generate and args.prompt is None:
        parser.error("--generate を使用する場合はプロンプトを指定してください。")
    return CliConfig(
        model=args.model,
        prompt=args.prompt,
        host=args.host.rstrip("/"),
        system_prompt=args.system_prompt,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        stream=not args.no_stream,
        as_json=args.json,
        show_payload=args.show_payload,
        use_generate=args.generate,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI のエントリーポイント。"""

    config = parse_args(argv)
    print(f"[orun] model: {config.model}", file=sys.stderr)
    try:
        if config.prompt is None and not config.use_generate:
            run_repl(config)
        else:
            run_once(config)
        return 0
    except OrunError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("処理が中断されました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
