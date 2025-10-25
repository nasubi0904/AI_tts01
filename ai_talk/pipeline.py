
"""会話処理パイプライン。

ASR → LLM → TTS → PLAY を低遅延で繋ぐため、最小限の同期処理と
2段ワーカー構成のみを残し、それ以外の待機は全てブロッキング待ちで
処理する。Busy loop を避けることで CPU スピンをなくし、スループット
とレスポンスの両面を改善する。
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Iterable

from .audio_player import AudioPlayer
from .config import OLLAMA_MODEL
from .llm_client import OllamaChatSession, describe_server
from .logger import Reporter, log
from .tts_voicevox import synthesize as tts_synth


_SENTINEL = object()


class TalkPipeline:
    """オーディオ応答パイプラインの調停役。"""

    def __init__(self, system_prompt: str = "", reporter: Reporter | None = None):
        self.reporter = reporter or Reporter()
        self.player = AudioPlayer(reporter=self.reporter)
        self.system_prompt = system_prompt
        self._chat = OllamaChatSession(system_prompt)
        self._input_q: "queue.Queue[str | object]" = queue.Queue()
        self._tts_q: "queue.Queue[str | object]" = queue.Queue()
        self._stop = threading.Event()
        self._llm_thr = threading.Thread(target=self._llm_worker, name="TalkLLM", daemon=True)
        self._tts_thr = threading.Thread(target=self._tts_worker, name="TalkTTS", daemon=True)
        self._llm_thr.start()
        self._tts_thr.start()
        self._log_server_status()

    # ------------------------------------------------------------------ public
    def push_user_text(self, text: str) -> None:
        """ユーザー入力をキューへ投入する。空文字列は無視。"""

        normalized = (text or "").strip()
        if not normalized:
            return
        prompt_payload = self._chat.compose_stream_payload(normalized)
        self.reporter.start_round(prompt_payload)
        self._input_q.put(normalized)

    def close(self) -> None:
        """全ワーカーを停止し、オーディオプレイヤーも閉じる。"""

        if self._stop.is_set():
            return
        self._stop.set()
        self._input_q.put(_SENTINEL)
        self._tts_q.put(_SENTINEL)
        self.player.stop()
        self._llm_thr.join(timeout=2)
        self._tts_thr.join(timeout=2)
        self.player.join(timeout=2)

    # ----------------------------------------------------------------- private
    def _log_server_status(self) -> None:
        """初期化時に Ollama サーバーの状態を記録する。"""

        info = describe_server()
        host = info.get("host", "(不明)")
        endpoint = info.get("endpoint", "(不明)")
        log(
            "INFO",
            f"Ollama接続設定 host={host} endpoint={endpoint} model={OLLAMA_MODEL}",
        )
        if info.get("reachable"):
            version = info.get("version") or "(不明)"
            models = info.get("models") or []
            summary = ", ".join(models[:3]) if models else "(モデル不明)"
            if len(models) > 3:
                summary += f" ... (+{len(models) - 3}件)"
            log("INFO", f"Ollama接続確認 version={version} models={summary}")
            if models and OLLAMA_MODEL not in models:
                log(
                    "WARN",
                    "現在の OLLAMA_MODEL がサーバー上に見つかりません。"
                    f" 直近の 404 はモデル未取得が原因の可能性があります。model={OLLAMA_MODEL}",
                )
        else:
            log("ERR", "Ollamaサーバーに接続できません。環境変数やエンドポイントを再確認してください。")

    def _llm_worker(self) -> None:
        """LLM 応答を取得し、文単位で TTS キューへ送る。"""

        while not self._stop.is_set():
            item = self._input_q.get()
            try:
                if item is _SENTINEL:
                    self._tts_q.put(_SENTINEL)
                    return
                assert isinstance(item, str)
                for sentence in self._iter_sentences(item):
                    self.reporter.llm_sentence(sentence)
                    self._tts_q.put(sentence)
            except Exception as exc:  # noqa: BLE001  runtime safety
                self.reporter.error("LLM", exc)
                traceback.print_exc()

    def _tts_worker(self) -> None:
        """テキストを音声化して再生キューへ投入する。"""

        while not self._stop.is_set():
            item = self._tts_q.get()
            try:
                if item is _SENTINEL:
                    return
                assert isinstance(item, str)
                wav = tts_synth(item)
                if wav:
                    self.reporter.tts_ready(item, len(wav))
                    self.player.enqueue({"wav": wav, "text": item})
            except Exception as exc:  # noqa: BLE001  runtime safety
                self.reporter.error("TTS", exc)
                traceback.print_exc()

    def _iter_sentences(self, user_text: str) -> Iterable[str]:
        """LLM ストリームから文を抽出して返す。"""

        for sent in self._chat.stream_sentences(user_text):
            normalized = sent.strip()
            if normalized:
                yield normalized
