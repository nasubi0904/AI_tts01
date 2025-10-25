
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

# Ollama へ渡す生成パラメータを細かく制御する。
OLLAMA_GENERATION_OPTIONS = {
    "temperature": 0.7,  # 創造性を司る温度。高いほど多様な応答になる。
    "top_p": 0.9,  # nucleus sampling。確率質量0.9に収まる語彙から選択。
    "top_k": 40,  # 上位k語から次の語を選ぶ。小さいほど保守的。
    "min_p": 0.05,  # 確率がこの値を下回る語を候補から除外する。
    "mirostat": 1,  # ミロスタット制御を有効化するフラグ。
    "mirostat_tau": 5.0,  # 目標エントロピー。出力の散らばりを制御。
    "mirostat_eta": 0.1,  # ミロスタットの学習率。収束の速さを調整。
    "num_ctx": 4096,  # 利用するコンテキスト長（トークン数）。
    "num_predict": 512,  # 応答の最大生成トークン数。
    "repeat_penalty": 1.1,  # 直近の語を繰り返さないように罰則を与える。
    "repeat_last_n": 64,  # 繰り返し抑制の対象とする直近トークン数。
    "seed": 42,  # 乱数シードを固定し応答の再現性を確保。
}

# ストップ語など、オプション以外のパラメータをまとめる。
OLLAMA_PAYLOAD_OVERRIDES = {
    "stop": ["\nUser:", "\nユーザー:"],  # 停止語は配列で複数指定できる。
}

def demo_keyboard(verbose=True, color=True):
    setup(verbose=verbose, color=color)
    rep = Reporter()
    tp = TalkPipeline(
        reporter=rep,
        llm_options=OLLAMA_GENERATION_OPTIONS,
        llm_payload_overrides=OLLAMA_PAYLOAD_OVERRIDES,
    )
    log("INFO", "対話モード。'exit' で終了。")
    try:
        while True:
            s = input("> ").strip()
            if s.lower() in {"exit", "quit"}:
                break
            if s:
                tp.push_user_text(s)
    finally:
        tp.close()

if __name__ == "__main__":
    demo_keyboard()
