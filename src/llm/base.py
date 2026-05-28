import threading
from abc import ABC


class BaseLlm(ABC):
    """Tracks per-thread last-call usage and aggregate usage across calls."""

    def __init__(self, cfg, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = cfg.max_tokens
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature

        self._usage_lock = threading.Lock()
        self._usage = {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._thread_local = threading.local()

        _orig = self.inference
        _self = self

        def _tracked_inference(*args, **kwargs):
            input_text = _self._extract_input_text(args, kwargs)
            prompt_tokens = len(_self.tokenizer.encode(input_text)) if input_text.strip() else 0
            result = _orig(*args, **kwargs)
            output_text = _self._extract_output_text(result)
            completion_tokens = len(_self.tokenizer.encode(output_text)) if output_text.strip() else 0
            with _self._usage_lock:
                _self._usage["call_count"] += 1
                _self._usage["prompt_tokens"] += prompt_tokens
                _self._usage["completion_tokens"] += completion_tokens
            _self._thread_local.last_usage = {
                "llm_call_count": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            return result

        self.inference = _tracked_inference

    def _extract_input_text(self, args, kwargs):
        messages = kwargs.get("messages")
        if messages:
            parts = []
            for m in messages:
                if isinstance(m, dict):
                    parts.append(str(m.get("content", "")))
                elif hasattr(m, "content"):
                    parts.append(str(m.content or ""))
            return " ".join(parts)
        sp = kwargs.get("system_prompt", "")
        up = kwargs.get("user_prompt", "")
        if not sp and not up and args:
            sp = str(args[0]) if len(args) > 0 else ""
            up = str(args[1]) if len(args) > 1 else ""
        return f"{sp} {up}"

    def _extract_output_text(self, result):
        if isinstance(result, str):
            return result
        if hasattr(result, "content"):
            return str(result.content or "")
        return str(result)

    def reset_usage(self):
        with self._usage_lock:
            self._usage = {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def get_usage(self):
        with self._usage_lock:
            return {
                "llm_call_count": self._usage["call_count"],
                "prompt_tokens": self._usage["prompt_tokens"],
                "completion_tokens": self._usage["completion_tokens"],
                "total_tokens": self._usage["prompt_tokens"] + self._usage["completion_tokens"],
            }

    def get_last_call_usage(self):
        return getattr(self._thread_local, "last_usage", {})

    def inference(self, system_prompt, user_prompt, max_new_tokens, temperature, **kwargs) -> str:
        raise NotImplementedError
