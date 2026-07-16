"""
Pipeline model registry — swap stages here without rewriting voice_loop.

Usage:
    from pipeline_config import DEFAULT_PIPELINE, build_handlers

    handlers = build_handlers(DEFAULT_PIPELINE, on_turn_complete=cb)

    # Or override one stage:
    build_handlers({**DEFAULT_PIPELINE, "llm": "qwen3_1_7b", "stt": "whisper_base"})

Available keys are listed in AVAILABLE.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Optional

from interfaces import BlockingLLMAdapter


ROOT = Path(__file__).resolve().parent

DEFAULT_PIPELINE = {
    # Default LLM is blocking Qwen3 — see LATENCY_NOTES.md (~0.5s faster Response Start vs streaming).
    "stt": "whisper_tiny",
    "ser": "wav2vec2_superb",
    "llm": "qwen3_1_7b",
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


def _make_llm_qwen():
    from llm import LLMHandler
    return LLMHandler()


def _make_llm_alternate(filename: str, label: str):
    module = _load_alternate_module(filename)
    return BlockingLLMAdapter(module.LLMHandler(), name=label)


def _make_tts_chatterbox(on_turn_complete=None):
    from tts import TTSHandler
    return TTSHandler(on_turn_complete=on_turn_complete)


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
    "qwen3_1_7b": _make_llm_qwen,
    # AlternateModels — blocking today, wrapped so voice_loop still works.
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
