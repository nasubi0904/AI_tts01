
"""
デモ：キーボード入力で対話。Reporter によるCLI可視化を有効化。
"""
import os
import sys

if __package__ is None or __package__ == "":
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    _PACKAGE_PARENT = os.path.dirname(_MODULE_DIR)
    if _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)

    from ai_talk.pipeline import TalkPipeline
    from ai_talk.logger import Reporter, setup, log
else:
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
