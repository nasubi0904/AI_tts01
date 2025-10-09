
"""
単一再生キュー（FIFO）。ファイルI/Oなし。
- キュー要素は bytes か {"wav": bytes, "text": str} を受け付ける。
- 再生開始時に reporter.play_start(text) を通知可能。
"""
import threading, queue, winsound
from .logger import Reporter, log

class AudioPlayer:
    def __init__(self, reporter: Reporter|None=None):
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._rep = reporter
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if isinstance(item, (bytes, bytearray)):
                    wav, text = bytes(item), ""
                elif isinstance(item, dict) and isinstance(item.get("wav"), (bytes, bytearray)):
                    wav, text = bytes(item["wav"]), str(item.get("text",""))
                else:
                    raise TypeError("enqueue expects bytes or {'wav':bytes,'text':str}")
                if self._rep:
                    self._rep.play_start(text)
                winsound.PlaySound(wav, winsound.SND_MEMORY)
            except Exception as e:
                log("ERR", f"PLAY error: {e}")
            finally:
                self._q.task_done()

    def enqueue(self, item):
        self._q.put(item)

    def flush(self):
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass

    def stop(self):
        self.flush()
        self._stop.set()

    def join(self, timeout=None):
        self._thr.join(timeout)
