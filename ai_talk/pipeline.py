
"""
ASR→LLM→TTS→PLAY の直列パイプライン。
- 文単位ストリームで先行TTS。
- ReporterでCLI可視化（LLM/TTS/PLAYの時刻を通知）。
"""
import threading, queue, traceback
from .audio_player import AudioPlayer
from .tts_voicevox import synthesize as tts_synth
from .llm_client import stream as llm_stream, describe_server
from .logger import Reporter, log

class TalkPipeline:
    def __init__(self, system_prompt: str = "", reporter: Reporter|None=None):
        self.reporter = reporter or Reporter()
        self.player = AudioPlayer(reporter=self.reporter)
        self.system_prompt = system_prompt
        self.in_q = queue.Queue()
        self._tts_q = queue.Queue()
        self._stop = threading.Event()
        self._llm_thr = threading.Thread(target=self._llm_worker, daemon=True)
        self._tts_thr = threading.Thread(target=self._tts_worker, daemon=True)
        self._llm_thr.start()
        self._tts_thr.start()
        # LLM 起動直後にサーバー情報を記録しておくと、接続不良時に原因特定が容易になる。
        info = describe_server()
        if info.get("reachable"):
            version = info.get("version") or "(不明)"
            models = info.get("models") or []
            summary = ", ".join(models[:3]) if models else "(モデル不明)"
            if len(models) > 3:
                summary += f" ... (+{len(models)-3}件)"
            log("INFO", f"Ollama接続確認 version={version} models={summary}")
        else:
            log("ERR", "Ollamaサーバーに接続できません。環境変数やエンドポイントを再確認してください。")

    def push_user_text(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        self.reporter.start_round(text)
        self.in_q.put(text)

    def _llm_worker(self):
        while not self._stop.is_set():
            try:
                user_text = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                for sent in llm_stream(prompt=user_text, system=self.system_prompt):
                    s = sent.strip()
                    if not s:
                        continue
                    self.reporter.llm_sentence(s)
                    self._tts_q.put(s)
            except Exception as e:
                self.reporter.error("LLM", e)
                traceback.print_exc()
            finally:
                self.in_q.task_done()

    def _tts_worker(self):
        while not self._stop.is_set():
            try:
                text = self._tts_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                wav = tts_synth(text)
                if wav:
                    self.reporter.tts_ready(text, len(wav))
                    self.player.enqueue({"wav": wav, "text": text})
            except Exception as e:
                self.reporter.error("TTS", e)
                traceback.print_exc()
            finally:
                self._tts_q.task_done()

    def close(self):
        self._stop.set()
        self.player.stop()
        self.player.join(timeout=2)
