"""Prompt loader — reads category JSON files bundled with the package."""

import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent
CATEGORIES = ["reasoning", "coding", "summarization", "creative", "qa"]


def load_prompts(categories: list[str] | None = None) -> list[dict]:
    """Load prompts for the given categories (default: all).

    Returns a list of dicts with keys: category, length, text.
    """
    selected = categories if categories is not None else CATEGORIES
    prompts = []
    for cat in selected:
        path = PROMPT_DIR / f"{cat}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        prompts.extend(json.loads(path.read_text()))
    return prompts
