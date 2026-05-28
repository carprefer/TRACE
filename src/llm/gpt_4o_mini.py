import logging
import os

import tiktoken
from openai import OpenAI

from .base import BaseLlm
from .registry import register_llm

logger = logging.getLogger(__name__)


@register_llm("gpt-4o-mini")
class Gpt4oMini(BaseLlm):
    """OpenAI gpt-4o-mini via Chat Completions API."""

    def __init__(self, cfg, acc=None):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        super().__init__(cfg, tokenizer)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm_name = cfg.name

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

        params = {
            "model": self.llm_name,
            "messages": messages,
            "max_completion_tokens": max_new_tokens if max_new_tokens else self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            **kwargs,
        }
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def validate_connection(self):
        self.client.models.list()
        logger.info("OpenAI gpt-4o-mini connection validated.")
