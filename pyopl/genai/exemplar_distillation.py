import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

from ._strategy_base import GenAIStrategyBase, LLMProvider

DESCRIPTION_MAX_CHARS = 1200
FEW_SHOT_COUNT = 3
_LOGGER = logging.getLogger(__name__)
_STRATEGY = GenAIStrategyBase(logger=_LOGGER, few_shot_top_k=FEW_SHOT_COUNT)


@dataclass(frozen=True)
class ExemplarDraft:
    name: str
    model: str
    data: str
    description: str
    source_session: str

    def with_reviewed_content(self, *, description: str, model: str, data: str) -> "ExemplarDraft":
        return replace(self, description=description, model=model, data=data)


def build_distillation_prompt(
    *,
    model: str,
    data: str,
    source_session: str,
    examples: Sequence[dict[str, str]],
) -> str:
    few_shots = _STRATEGY.render_few_shots_section(list(examples))
    return (
        "Write only a concise plain-text description of the current optimization problem.\n"
        "Describe the decisions, constraints, objective, and important data relationships. "
        "Do not reproduce OPL syntax or mention the session, an LLM, examples, or prompting. "
        "Generalize implementation-specific names where practical. "
        f"Keep the response within {DESCRIPTION_MAX_CHARS} characters.\n\n"
        "The examples below are style references only. Do not copy their problem details or let "
        "them change the meaning of the current model, data, and session information.\n\n"
        f"{few_shots}"
        f"<current_model>\n{model}\n</current_model>\n\n"
        f"<current_data>\n{data}\n</current_data>\n\n"
        f"<current_session>\n{source_session}\n</current_session>\n"
    )


def normalize_description(response: str) -> str:
    text = response.strip()
    fenced = re.fullmatch(r"```(?:text|plaintext|markdown)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(
        r"^\s*(?:problem\s+description|description|answer)\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    if len(text) > DESCRIPTION_MAX_CHARS:
        text = text[:DESCRIPTION_MAX_CHARS].rstrip()
    return text


def distill_exemplar_description(
    *,
    model: str,
    data: str,
    source_session: str,
    provider: str,
    model_name: str,
    models_dir: Optional[str | Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    provider_key = provider.strip().lower()
    try:
        selected_provider = LLMProvider(provider_key)
    except ValueError as exc:
        raise ValueError(f"Unsupported GenAI provider: {provider}") from exc

    retrieval_query = "\n\n".join(part for part in (source_session, model, data) if part.strip())
    examples = _STRATEGY.gather_few_shots(
        retrieval_query,
        k=FEW_SHOT_COUNT,
        models_dir=models_dir,
        progress=progress,
    )
    if len(examples) < FEW_SHOT_COUNT:
        message = f"Exemplar description: using {len(examples)} of {FEW_SHOT_COUNT} requested style examples."
        _LOGGER.warning(message)
        if progress:
            progress(message)

    prompt = build_distillation_prompt(
        model=model,
        data=data,
        source_session=source_session,
        examples=examples,
    )
    response = _STRATEGY.llm_generate_text(
        provider=selected_provider,
        model_name=model_name,
        input_text=prompt,
        max_tokens=500,
        temperature=0.2,
    )
    description = normalize_description(str(response))
    if not description:
        raise RuntimeError("The GenAI model returned an empty exemplar description.")
    return description
