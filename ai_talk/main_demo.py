
"""
デモ：キーボード入力で対話。Reporter によるCLI可視化を有効化。
"""
import time
from .pipeline import TalkPipeline
from .logger import Reporter, setup, log

SYSTEM_PROMPT = "日本語で簡潔に答える。"

def demo_keyboard(verbose=True, color=True):
    setup(verbose=verbose, color=color)
    rep = Reporter()
    tp = TalkPipeline(system_prompt=SYSTEM_PROMPT, reporter=rep)
    log("INFO", "対話モード。'exit' で終了。")
    try:
        while True:
            s = input("> ").strip()
            if s.lower() in {"exit","quit"}: break
            if s: tp.push_user_text(s)
    finally:
        tp.close()

if __name__ == "__main__":
    demo_keyboard()
