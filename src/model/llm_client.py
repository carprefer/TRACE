"""LLM client factory backed by ../llm/ and central config.LLM_PRESETS."""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so `src.llm`/`src.config` resolve

from src import config
from src.llm import load_llm


def make_client(model: str | None = None, **overrides):
    name = model or config.LLM
    if name not in config.LLM_PRESETS:
        raise ValueError(
            f"unknown model preset {name!r}; add to config.LLM_PRESETS. "
            f"Available: {sorted(config.LLM_PRESETS.keys())}"
        )
    cfg_dict = dict(config.LLM_PRESETS[name])
    cfg_dict.update(overrides)
    return load_llm(OmegaConf.create(cfg_dict))
