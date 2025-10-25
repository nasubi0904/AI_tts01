
"""簡易ロガーと処理時間計測レポーター。

責務分離・可読性向上・冗長性排除を目的に、ログ出力機構とレポート機能を
明確に分割している。VSCode ターミナルなど ANSI 対応環境での色付き出力に
対応する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sys
import time
from typing import Final, Mapping, Sequence, TextIO


# ---------------------------------------------------------------------------
# 設定および色付けロジック


DEFAULT_PALETTE: Final[Mapping[str, str]] = {
    "RESET": "\033[0m",
    "INFO": "\033[36m",
    "PROMPT": "\033[36m",
    "LLM": "\033[35m",
    "TTS": "\033[33m",
    "PLAY": "\033[32m",
    "ASR": "\033[34m",
    "ERR": "\033[31m",
}


@dataclass(slots=True)
class LoggerConfig:
    """ログ出力の挙動を司る設定値。"""

    verbose: bool = True
    enable_color: bool = True

    def color_enabled(self) -> bool:
        """NO_COLOR 環境変数を考慮した色出力可否を返す。"""

        return self.enable_color and (os.getenv("NO_COLOR", "") == "")

    def should_emit(self, tag: str) -> bool:
        """指定タグを出力対象とするか判定する。"""

        return self.verbose or tag == "ERR"


@dataclass(slots=True)
class AnsiColorFormatter:
    """ANSI カラーコードを用いてログ行を整形するフォーマッタ。"""

    palette: Mapping[str, str] = field(default_factory=lambda: DEFAULT_PALETTE)

    def apply(self, tag: str, message: str) -> str:
        """タグに応じた色付けを適用した文字列を返す。"""

        color = self.palette.get(tag, "")
        reset = self.palette.get("RESET", "")
        return f"{color}{message}{reset}" if color and reset else message


@dataclass(slots=True)
class ConsoleLogger:
    """コンソールへログを出力する軽量ロガー。"""

    config: LoggerConfig = field(default_factory=LoggerConfig)
    formatter: AnsiColorFormatter = field(default_factory=AnsiColorFormatter)
    stream: TextIO = field(default=sys.stdout, repr=False)

    def log(self, tag: str, message: str) -> None:
        """タグ付きメッセージを出力する。"""

        if not self.config.should_emit(tag):
            return
        timestamp = time.strftime("%H:%M:%S")
        base = f"[{timestamp} {tag}] {message}"
        rendered = base
        if self.config.color_enabled():
            rendered = self.formatter.apply(tag, base)
        self.stream.write(rendered + "\n")
        self.stream.flush()


# ---------------------------------------------------------------------------
# グローバルエントリポイント


_GLOBAL_CONFIG = LoggerConfig()
_GLOBAL_LOGGER = ConsoleLogger(config=_GLOBAL_CONFIG)


def setup(*, verbose: bool = True, color: bool = True) -> None:
    """グローバルなログ設定を更新する。"""

    _GLOBAL_CONFIG.verbose = verbose
    _GLOBAL_CONFIG.enable_color = color


def log(tag: str, message: str) -> None:
    """グローバルロガーを利用してメッセージを出力する。"""

    _GLOBAL_LOGGER.log(tag, message)


# ---------------------------------------------------------------------------
# レポーター: 処理時間やイベントの可視化


@dataclass(slots=True)
class Reporter:
    """各処理段階のタイムスタンプを記録しログへ通知する。"""

    logger: ConsoleLogger = field(default_factory=lambda: _GLOBAL_LOGGER)
    _started_at: float | None = field(default=None, init=False, repr=False)
    _first_token: float | None = field(default=None, init=False, repr=False)
    _first_tts: float | None = field(default=None, init=False, repr=False)
    _first_play: float | None = field(default=None, init=False, repr=False)

    def start_round(self, prompt_payload: Mapping[str, object] | Sequence[Mapping[str, str]] | None) -> None:
        """新しいユーザー入力処理を開始したことを記録する。"""

        self._started_at = time.perf_counter()
        self._first_token = self._first_tts = self._first_play = None
        payload_object: object
        if isinstance(prompt_payload, Mapping):
            payload_object = prompt_payload
        elif isinstance(prompt_payload, Sequence):
            payload_object = list(prompt_payload)
        else:
            payload_object = ""
        try:
            serialized = json.dumps(payload_object, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            serialized = repr(payload_object)
        self.logger.log("PROMPT", serialized)

    def llm_sentence(self, text: str) -> None:
        """LLM から文を受領した際の計測とログ出力。"""

        now = time.perf_counter()
        if self._first_token is None and self._started_at is not None:
            self._first_token = now - self._started_at
            self.logger.log("LLM", f"first_sentence {self._first_token * 1000:.0f} ms")
        self.logger.log("LLM", text)

    def tts_ready(self, text: str, nbytes: int) -> None:
        """TTS 音声生成完了を記録し、バイト数などを報告する。"""

        now = time.perf_counter()
        if self._first_tts is None and self._started_at is not None:
            self._first_tts = now - self._started_at
        _ = text, nbytes  # 将来の拡張に備え、引数は維持する

    def play_start(self, text: str) -> None:
        """音声再生開始時の遅延を計測しログ出力する。"""

        now = time.perf_counter()
        if self._first_play is None and self._started_at is not None:
            self._first_play = now - self._started_at
        _ = text  # ログ抑制のため未使用扱い

    def error(self, scope: str, exc: Exception) -> None:
        """処理中に発生した例外情報を記録する。"""

        self.logger.log("ERR", f"{scope}: {exc}")


__all__ = ["ConsoleLogger", "Reporter", "setup", "log"]
