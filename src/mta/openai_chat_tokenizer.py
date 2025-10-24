from __future__ import annotations

from typing import List, Sequence


class OpenAIChatTokenizer:
    """
    Minimal tokenizer interface used when routing LLM calls through OpenAI-compatible
    HTTP APIs. The execution engine only relies on ``encode`` and ``apply_chat_template``
    for bookkeeping, so a lightweight implementation suffices.
    """

    def __init__(self, model_name: str = "openai"):
        self.name_or_path = model_name
        self.bos_token = ""
        self.eos_token = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # Use UTF-8 bytes as a cheap proxy for token count
        return list(text.encode("utf-8"))

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: object,
    ):
        prompt_parts: list[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            prompt_parts.append(f"{role}:\n{content}\n\n")
        prompt = "".join(prompt_parts)
        if add_generation_prompt:
            prompt += "ASSISTANT:\n"
        if tokenize:
            return self.encode(prompt, add_special_tokens=False)
        return prompt
