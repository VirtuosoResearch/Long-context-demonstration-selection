from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

PROMPTS_DIR = Path(__file__).resolve().parent


def available_languages() -> Iterable[str]:
    for file in PROMPTS_DIR.glob("*.json"):
        yield file.stem.lower()


def load_prompt(language: str) -> str | None:
    prompt_file = PROMPTS_DIR / f"{language.lower()}.json"
    if not prompt_file.exists():
        return None
    try:
        with prompt_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    prompt = data.get("prompt")
    return prompt if isinstance(prompt, str) else None


def resolve_prompt(language: str, *, default: str | None = None) -> str | None:
    language_prompt = load_prompt(language)
    if language_prompt:
        return language_prompt
    return default


__all__ = ["available_languages", "load_prompt", "resolve_prompt"]
