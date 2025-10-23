
"""非同期オーディオ再生ワーカー。"""

from __future__ import annotations

import queue
import threading
import winsound
from typing import Any

from .logger import Reporter, log


_SENTINEL = object()


class AudioPlayer:
    """FIFO キューから WAV データを順次再生する。"""

    def __init__(self, reporter: Reporter | None = None):
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._stop = threading.Event()
        self._rep = reporter
        self._thr = threading.Thread(target=self._worker, name="AudioPlayer", daemon=True)
        self._thr.start()

    def enqueue(self, item: Any) -> None:
        """再生キューへ追加する。"""

        self._q.put(item)

    def stop(self) -> None:
        """待機中のワーカーを停止させる。"""

        if self._stop.is_set():
            return
        self._stop.set()
        self._q.put(_SENTINEL)

    def join(self, timeout: float | None = None) -> None:
        """ワーカー終了を待機する。"""

        self._thr.join(timeout)

    # ----------------------------------------------------------------- worker
    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _SENTINEL:
                    return
                wav, text = self._normalize_item(item)
                if self._rep:
                    self._rep.play_start(text)
                winsound.PlaySound(wav, winsound.SND_MEMORY)
            except Exception as exc:  # noqa: BLE001  runtime safety
                log("ERR", f"PLAY error: {exc}")

    @staticmethod
    def _normalize_item(item: Any) -> tuple[bytes, str]:
        if isinstance(item, (bytes, bytearray)):
            return bytes(item), ""
        if isinstance(item, dict) and isinstance(item.get("wav"), (bytes, bytearray)):
            text = str(item.get("text", ""))
            return bytes(item["wav"]), text
        raise TypeError("enqueue expects bytes or {'wav': bytes, 'text': str}")
