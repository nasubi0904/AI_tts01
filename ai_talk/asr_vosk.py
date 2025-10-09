
"""
Vosk ASR。確定結果を callback で渡す。
"""
import json, threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from .config import ASR_DEVICE, ASR_BLOCK_SIZE, ASR_SAMPLE_RATE
from .logger import log

class VoskASR:
    def __init__(self, model_path: str, on_final):
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, ASR_SAMPLE_RATE)
        self.on_final = on_final
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        if self._thr and self._thr.is_alive(): return
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def _worker(self):
        device = None if ASR_DEVICE == "" else ASR_DEVICE
        with sd.RawInputStream(samplerate=ASR_SAMPLE_RATE, blocksize=ASR_BLOCK_SIZE, device=device, dtype="int16", channels=1) as stream:
            while not self._stop.is_set():
                data = stream.read(ASR_BLOCK_SIZE)[0]
                if len(data)==0: continue
                if self.rec.AcceptWaveform(data):
                    res = json.loads(self.rec.Result())
                    text = res.get("text","").strip()
                    if text:
                        log("ASR", f"final: {text}")
                        self.on_final(text)

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)
