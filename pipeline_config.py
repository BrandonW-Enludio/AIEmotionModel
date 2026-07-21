"""
Pipeline model registry — swap stages here without rewriting voice_loop.

Usage:
    from pipeline_config import DEFAULT_PIPELINE, build_handlers

    handlers = build_handlers(DEFAULT_PIPELINE, on_turn_complete=cb)

    # Or override one stage:
    #   build_handlers({**DEFAULT_PIPELINE, "llm": "openai_compat"})
    # openai_compat needs OPENAI_MODEL (+ optional OPENAI_BASE_URL / OPENAI_API_KEY).

Available keys are listed in AVAILABLE.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Optional

from interfaces import BlockingLLMAdapter


ROOT = Path(__file__).resolve().parent

DEFAULT_PIPELINE = {
    # LLM via llama.cpp / any OpenAI-compat server (see llm_openai.py).
    # Swap back to "qwen2_5_7b" for in-process HuggingFace.
    "stt": "whisper_tiny",
    "ser": "wav2vec2_superb",
    "llm": "openai_compat",
    "tts": "chatterbox_turbo",
}


def _load_alternate_module(filename: str):
    path = ROOT / "AlternateModels" / filename
    if not path.exists():
        raise FileNotFoundError(f"Alternate model file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_stt(model_size: str):
    from stt import STTHandler
    return STTHandler(model_size=model_size)


def _make_ser_default():
    from ser import SERHandler
    return SERHandler()


def _make_llm_qwen3_1_7b():
    from llm import LLMHandler
    from scenario import ACTIVE_SCENARIO_ID
    return LLMHandler(
        scenario_id=ACTIVE_SCENARIO_ID,
        model_name="Qwen/Qwen3-1.7B",
        display_name="Qwen3 1.7B",
    )


def _make_llm_qwen2_5_7b():
    from llm import LLMHandler
    from scenario import ACTIVE_SCENARIO_ID
    return LLMHandler(
        scenario_id=ACTIVE_SCENARIO_ID,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5-7B-Instruct",
    )


def _make_llm_openai_compat():
    from llm_openai import OpenAICompatLLMHandler
    from scenario import ACTIVE_SCENARIO_ID
    return OpenAICompatLLMHandler(scenario_id=ACTIVE_SCENARIO_ID)


def _make_llm_alternate(filename: str, label: str):
    module = _load_alternate_module(filename)
    return BlockingLLMAdapter(module.LLMHandler(), name=label)


def _make_tts_chatterbox(on_turn_complete=None):
    from tts import TTSHandler
    from scenario import ACTIVE_SCENARIO_ID
    return TTSHandler(
        on_turn_complete=on_turn_complete,
        scenario_id=ACTIVE_SCENARIO_ID,
    )


STT_REGISTRY: dict[str, Callable] = {
    "whisper_tiny": lambda: _make_stt("tiny"),
    "whisper_base": lambda: _make_stt("base"),
    "whisper_small": lambda: _make_stt("small"),
    "whisper_medium": lambda: _make_stt("medium"),
}

SER_REGISTRY: dict[str, Callable] = {
    "wav2vec2_superb": _make_ser_default,
}

LLM_REGISTRY: dict[str, Callable] = {
    "qwen3_1_7b": _make_llm_qwen3_1_7b,
    "qwen2_5_7b": _make_llm_qwen2_5_7b,
    # Any OpenAI-style /v1/chat/completions (llama.cpp, cloud, etc.).
    "openai_compat": _make_llm_openai_compat,
    # AlternateModels — own prompts today; do not use scenario.py yet.
    "gemma2_2b": lambda: _make_llm_alternate("llm_gemma2B.py", "gemma2_2b"),
    "gemma4_12b": lambda: _make_llm_alternate("llm_gemma4_12B.py", "gemma4_12b"),
    "phi3_5_mini": lambda: _make_llm_alternate("llm_Phi3_5_mini.py", "phi3_5_mini"),
}

TTS_REGISTRY: dict[str, Callable] = {
    "chatterbox_turbo": _make_tts_chatterbox,
    # Qwen TTS lives in AlternateModels but needs a TTSInterface adapter first.
}

AVAILABLE = {
    "stt": sorted(STT_REGISTRY.keys()),
    "ser": sorted(SER_REGISTRY.keys()),
    "llm": sorted(LLM_REGISTRY.keys()),
    "tts": sorted(TTS_REGISTRY.keys()),
}


def _resolve(registry: dict, key: str, stage: str):
    if key not in registry:
        raise KeyError(
            f"Unknown {stage} model '{key}'. Available: {sorted(registry.keys())}"
        )
    return registry[key]


def build_handlers(
    config: Optional[dict] = None,
    on_turn_complete=None,
):
    """
    Build STT / SER / LLM / TTS handlers from a config dict.

    Returns dict with keys: stt, ser, llm, tts, config
    """
    cfg = {**DEFAULT_PIPELINE, **(config or {})}

    stt = _resolve(STT_REGISTRY, cfg["stt"], "stt")()
    ser = _resolve(SER_REGISTRY, cfg["ser"], "ser")()
    llm = _resolve(LLM_REGISTRY, cfg["llm"], "llm")()
    tts_factory = _resolve(TTS_REGISTRY, cfg["tts"], "tts")
    tts = tts_factory(on_turn_complete=on_turn_complete)

    return {
        "stt": stt,
        "ser": ser,
        "llm": llm,
        "tts": tts,
        "config": cfg,
    }


def print_available():
    print("Available pipeline models:")
    for stage, names in AVAILABLE.items():
        print(f"  {stage}: {', '.join(names)}")
