import logging
import os
import re

import tiktoken
from openai import OpenAI

from .base import BaseLlm
from .registry import register_llm

logger = logging.getLogger(__name__)


@register_llm("qwen3.5-35b-a3b")
class Qwen3_5(BaseLlm):
    """vLLM-served Qwen3.5-35B-A3B. Strips <think>...</think> spans."""

    def __init__(self, cfg, acc=None):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        super().__init__(cfg, tokenizer)
        base_url = cfg.get("base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
        api_key = cfg.get("api_key", os.environ.get("VLLM_API_KEY", "EMPTY"))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.llm_name = cfg.get("served_model_name", cfg.name)
        self.thinking_budget = cfg.get("thinking_budget", 0)

    def inference(
        self,
        messages=None,
        system_prompt="",
        user_prompt="",
        max_new_tokens=None,
        temperature=None,
        **kwargs,
    ):
        if not messages:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

        max_tokens = max_new_tokens if max_new_tokens else self.max_new_tokens
        enable_thinking = self.thinking_budget > 0

        params = {
            "model": self.llm_name,
            "messages": messages,
            "max_tokens": max_tokens + self.thinking_budget,
            "temperature": temperature if temperature is not None else self.temperature,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                    "thinking_budget": self.thinking_budget,
                },
            },
            **kwargs,
        }
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if "<think>" in content:
                content = content.split("<think>")[0].strip()
        return content

    def validate_connection(self):
        self.client.models.list()
        logger.info("vLLM Qwen3.5-35B-A3B connection validated.")
