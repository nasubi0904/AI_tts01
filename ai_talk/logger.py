
"""
簡易ロガー＋レポーター。ANSI色つき。VSCodeターミナル対応。
"""
import sys, time, os

COLORS = {
    "RESET": "\033[0m",
    "DIM": "\033[2m",
    "BOLD": "\033[1m",
    "INFO": "\033[36m",
    "LLM": "\033[35m",
    "TTS": "\033[33m",
    "PLAY": "\033[32m",
    "ASR": "\033[34m",
    "ERR": "\033[31m",
}

_VERBOSE = True
_COLOR = True

def setup(verbose: bool=True, color: bool=True):
    global _VERBOSE, _COLOR
    _VERBOSE = verbose
    _COLOR = color and (os.getenv("NO_COLOR","")=="")

def _c(tag):
    return COLORS.get(tag, "")

def log(tag: str, msg: str):
    if not _VERBOSE and tag not in ("ERR",):
        return
    ts = time.strftime("%H:%M:%S")
    if _COLOR:
        sys.stdout.write(f"{_c(tag)}[{ts} {tag}] {msg}{_c('RESET')}\n")
    else:
        sys.stdout.write(f"[{ts} {tag}] {msg}\n")
    sys.stdout.flush()

class Reporter:
    def __init__(self):
        self.t0 = None
        self.first_token = None
        self.first_tts = None
        self.first_play = None

    def start_round(self, prompt:str):
        self.t0 = time.perf_counter()
        self.first_token = self.first_tts = self.first_play = None
        log("INFO", f"PROMPT: {prompt}")

    def llm_sentence(self, text:str):
        now = time.perf_counter()
        if self.first_token is None and self.t0 is not None:
            self.first_token = now - self.t0
            log("LLM", f"first_sentence {self.first_token*1000:.0f} ms")
        log("LLM", f"{text}")

    def tts_ready(self, text:str, nbytes:int):
        now = time.perf_counter()
        if self.first_tts is None and self.t0 is not None:
            self.first_tts = now - self.t0
            log("TTS", f"first_audio_ready {self.first_tts*1000:.0f} ms")
        log("TTS", f"queued bytes={nbytes} text='{text[:24]}'")

    def play_start(self, text:str):
        now = time.perf_counter()
        if self.first_play is None and self.t0 is not None:
            self.first_play = now - self.t0
            log("PLAY", f"first_play {self.first_play*1000:.0f} ms")
        log("PLAY", f"start '{text[:24]}'")

    def error(self, scope:str, e:Exception):
        log("ERR", f"{scope}: {e}")
