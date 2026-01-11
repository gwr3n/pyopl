"""
Provider model discovery helpers.

This is a stable facade over internal shared implementations in _strategy_base.
"""

from __future__ import annotations

from typing import Optional

from ._strategy_base import (
    list_gemini_models as _list_gemini_models,
)
from ._strategy_base import (
    list_models as _list_models,
)
from ._strategy_base import (
    list_ollama_models as _list_ollama_models,
)
from ._strategy_base import (
    list_openai_models as _list_openai_models,
)


def list_openai_models(prefix: Optional[str] = "gpt") -> list[str]:
    return _list_openai_models(prefix=prefix)


def list_gemini_models(prefix: Optional[str] = "gemini") -> list[str]:
    return _list_gemini_models(prefix=prefix)


def list_ollama_models(prefix: Optional[str] = None) -> list[str]:
    return _list_ollama_models(prefix=prefix)


def list_models(llm_provider: Optional[str] = None, model_name: str = "gpt-5") -> list[str]:
    return _list_models(llm_provider=llm_provider, model_name=model_name)
