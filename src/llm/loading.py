from . import gpt_4o_mini, qwen3_5  # noqa: F401 — registers @register_llm
from .registry import get_llm_cls


def load_llm(cfg, acc=None):
    llm_cls = get_llm_cls(cfg.name)
    return llm_cls(cfg, acc)
