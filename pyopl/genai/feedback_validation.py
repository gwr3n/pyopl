"""Canonical generation and validation of feedback-proposed PyOPL revisions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ..pyopl_core import OPLCompiler
from ._strategy_base import GenAIStrategyBase, Grammar, ImageInput, PromptInput

MAX_ITERATIONS = 5
MAX_OUTPUT_TOKENS = None
MODEL_NAME = "gpt-5"
LLM_PROVIDER = "openai"

logger = logging.getLogger(__name__)
_BASE = GenAIStrategyBase(logger=logger, max_output_tokens=MAX_OUTPUT_TOKENS)


def _notify(progress: Optional[Callable[[str], None]], message: str) -> None:
    _BASE.notify(progress, message)


def _grammar_mode(mode: Any) -> Grammar:
    name = getattr(mode, "name", None)
    if isinstance(name, str) and name in Grammar.__members__:
        return Grammar[name]
    if isinstance(mode, Grammar):
        return mode
    return Grammar.BNF


def _build_feedback_prompt(question: str, grammar: str, model: str, data: str) -> str:
    guidelines = (
        "When revised content is necessary, label objective and constraints. "
        "Include concise comments explaining variables, parameters, and constraints, "
        "aligned to the user's question and the problem (literate style).\n"
    )

    return (
        "<role>\nYou are an expert in mathematical optimization and PyOPL.\n</role>\n\n"
        "<task>\nAnswer the user's question about the provided PyOPL model and data. "
        "Provide critical, specific feedback. If revisions are necessary, propose the minimal changes.\n"
        f"{guidelines}"
        "Do not change existing parts of the model and data unless strictly necessary.\n"
        "</task>\n\n"
        f"<grammar_reference>\n{grammar}\n</grammar_reference>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<model>\n{model}\n</model>\n\n"
        f"<data>\n{data}\n</data>\n\n"
        "<output_requirements>\n"
        'Return only a JSON object with required string key "feedback" and optional '
        'string keys "revised_model" and "revised_data". Return complete file contents, not diffs.\n'
        "</output_requirements>"
    )


def _build_alignment_prompt(
    question: str,
    grammar: str,
    original_model: str,
    original_data: str,
    candidate_model: str,
    candidate_data: str,
) -> str:
    return (
        "<role>\nYou validate proposed revisions to a PyOPL model and data.\n</role>\n\n"
        "<task>\nDetermine whether the candidate addresses the user's request, remains mutually consistent, "
        "and preserves unrelated semantics from the originals.\n</task>\n\n"
        f"<grammar_reference>\n{grammar}\n</grammar_reference>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<original_model>\n{original_model}\n</original_model>\n\n"
        f"<original_data>\n{original_data}\n</original_data>\n\n"
        f"<candidate_model>\n{candidate_model}\n</candidate_model>\n\n"
        f"<candidate_data>\n{candidate_data}\n</candidate_data>\n\n"
        '<output_requirements>\nReturn only JSON with exactly "aligned" (boolean) and '
        '"assessment" (string).\n</output_requirements>'
    )


def _build_repair_prompt(
    question: str,
    grammar: str,
    original_model: str,
    original_data: str,
    candidate_model: str,
    candidate_data: str,
    issue: str,
) -> str:
    guidelines = (
        "When revised content is necessary, label objective and constraints. "
        "Include concise comments explaining variables, parameters, and constraints, "
        "aligned to the user's question and the problem (literate style).\n"
    )
    
    return (
        "<role>\nYou repair proposed PyOPL model and data revisions.\n</role>\n\n"
        "<task>\nReturn the complete corrected candidate model and data. Make only changes needed to resolve "
        "the validation issue while satisfying the user's request and preserving unrelated behavior.\n"
        f"{guidelines}"
        "Do not change existing parts of the model and data unless strictly necessary.\n</task>\n\n"
        f"<grammar_reference>\n{grammar}\n</grammar_reference>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<original_model>\n{original_model}\n</original_model>\n\n"
        f"<original_data>\n{original_data}\n</original_data>\n\n"
        f"<candidate_model>\n{candidate_model}\n</candidate_model>\n\n"
        f"<candidate_data>\n{candidate_data}\n</candidate_data>\n\n"
        f"<validation_issue>\n{issue}\n</validation_issue>\n\n"
        '<output_requirements>\nReturn only JSON with exactly two string keys: "model" and "data".\n'
        "</output_requirements>"
    )


def _generate_json(
    *,
    provider: Any,
    model_name: str,
    prompt: str,
    images: list[ImageInput],
    temperature: Optional[float],
    stop: Optional[list[str]],
    progress: Optional[Callable[[str], None]],
) -> dict[str, Any]:
    content = _BASE.llm_generate_text(
        provider=provider,
        model_name=model_name,
        input_text=prompt,
        images=images,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        stop=stop,
        progress=progress,
        capture_usage=False,
        expected_json=True,
    )
    if not isinstance(content, str) or not content:
        raise RuntimeError("Empty model response.")
    parsed = _BASE.json_loads_relaxed(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response must be a JSON object.")
    return parsed


def _string_field(payload: dict[str, Any], key: str, *, required: bool = False) -> str:
    value = payload.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        qualifier = "non-empty " if required else ""
        raise RuntimeError(f"Feedback response field '{key}' must be a {qualifier}string.")
    return value


def _maybe_unescape(value: str) -> str:
    if "\\n" not in value or "\n" in value:
        return value
    try:
        return value.encode("utf-8").decode("unicode_escape")
    except Exception:
        return value.replace("\\n", "\n").replace("\\t", "\t")


def _repair_candidate(
    *,
    question: str,
    grammar: str,
    original_model: str,
    original_data: str,
    candidate_model: str,
    candidate_data: str,
    issue: str,
    provider: Any,
    model_name: str,
    images: list[ImageInput],
    temperature: Optional[float],
    stop: Optional[list[str]],
    progress: Optional[Callable[[str], None]],
) -> tuple[str, str]:
    repaired = _generate_json(
        provider=provider,
        model_name=model_name,
        prompt=_build_repair_prompt(
            question,
            grammar,
            original_model,
            original_data,
            candidate_model,
            candidate_data,
            issue,
        ),
        images=images,
        temperature=temperature,
        stop=stop,
        progress=progress,
    )
    return _string_field(repaired, "model", required=True), _string_field(repaired, "data", required=True)


def _validate_revisions(
    *,
    question: str,
    grammar: str,
    original_model: str,
    original_data: str,
    candidate_model: str,
    candidate_data: str,
    provider: Any,
    model_name: str,
    images: list[ImageInput],
    temperature: Optional[float],
    stop: Optional[list[str]],
    progress: Optional[Callable[[str], None]],
    alignment_check: bool,
    validation_iterations: int,
    syntax_error_reporting: str,
) -> tuple[Optional[tuple[str, str]], str]:
    last_issue = "Revision validation did not complete."
    for attempt in range(1, validation_iterations + 1):
        _notify(progress, f"Validating proposed revisions ({attempt}/{validation_iterations})")
        try:
            OPLCompiler(syntax_error_reporting=syntax_error_reporting).compile_model(candidate_model, candidate_data)
        except Exception as exc:
            last_issue = f"Compilation failed: {exc}"
        else:
            if not alignment_check:
                return (candidate_model, candidate_data), ""
            try:
                alignment = _generate_json(
                    provider=provider,
                    model_name=model_name,
                    prompt=_build_alignment_prompt(
                        question,
                        grammar,
                        original_model,
                        original_data,
                        candidate_model,
                        candidate_data,
                    ),
                    images=images,
                    temperature=temperature,
                    stop=stop,
                    progress=progress,
                )
                if not isinstance(alignment.get("aligned"), bool) or not isinstance(alignment.get("assessment"), str):
                    raise RuntimeError("Alignment response requires boolean 'aligned' and string 'assessment'.")
                if alignment["aligned"]:
                    return (candidate_model, candidate_data), ""
                last_issue = f"Alignment failed: {alignment['assessment'].strip()}"
            except Exception as exc:
                last_issue = f"Alignment validation failed: {exc}"

        if attempt >= validation_iterations:
            break
        _notify(progress, f"Repairing proposed revisions: {last_issue}")
        try:
            candidate_model, candidate_data = _repair_candidate(
                question=question,
                grammar=grammar,
                original_model=original_model,
                original_data=original_data,
                candidate_model=candidate_model,
                candidate_data=candidate_data,
                issue=last_issue,
                provider=provider,
                model_name=model_name,
                images=images,
                temperature=temperature,
                stop=stop,
                progress=progress,
            )
        except Exception as exc:
            last_issue = f"Revision repair failed: {exc}"
            break
    return None, last_issue


def generative_feedback(
    prompt: PromptInput,
    model_file: str | Path,
    data_file: str | Path,
    model_name: str = MODEL_NAME,
    mode: Any = Grammar.BNF,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    llm_provider: Optional[str] = LLM_PROVIDER,
    progress: Optional[Callable[[str], None]] = None,
    *,
    validate_revisions: bool = True,
    alignment_check: bool = True,
    validation_iterations: int = MAX_ITERATIONS,
    syntax_error_reporting: str = "full",
) -> dict[str, str]:
    """Return feedback and expose only revisions that pass compilation and alignment validation."""
    question, images = _BASE.normalize_prompt_input(prompt)
    original_model = Path(model_file).read_text(encoding="utf-8")
    original_data = Path(data_file).read_text(encoding="utf-8")
    grammar = _BASE.get_grammar_implementation(_grammar_mode(mode))
    provider = _BASE.infer_provider(llm_provider, model_name)
    deterministic_temperature = 0.0 if temperature is not None else None

    _notify(progress, "Generating feedback from LLM")
    payload = _generate_json(
        provider=provider,
        model_name=model_name,
        prompt=_build_feedback_prompt(question, grammar, original_model, original_data),
        images=images,
        temperature=deterministic_temperature,
        stop=stop,
        progress=progress,
    )
    feedback = _maybe_unescape(_string_field(payload, "feedback", required=True))
    revised_model = _maybe_unescape(_string_field(payload, "revised_model"))
    revised_data = _maybe_unescape(_string_field(payload, "revised_data"))
    result = {"feedback": feedback}

    if not revised_model and not revised_data:
        return result
    if not validate_revisions:
        if revised_model:
            result["revised_model"] = revised_model
        if revised_data:
            result["revised_data"] = revised_data
        return result

    try:
        attempts = max(1, int(validation_iterations))
    except (TypeError, ValueError):
        attempts = MAX_ITERATIONS
    candidate_model = revised_model or original_model
    candidate_data = revised_data or original_data
    validated, issue = _validate_revisions(
        question=question,
        grammar=grammar,
        original_model=original_model,
        original_data=original_data,
        candidate_model=candidate_model,
        candidate_data=candidate_data,
        provider=provider,
        model_name=model_name,
        images=images,
        temperature=deterministic_temperature,
        stop=stop,
        progress=progress,
        alignment_check=alignment_check,
        validation_iterations=attempts,
        syntax_error_reporting=syntax_error_reporting,
    )
    if validated is None:
        result["feedback"] = f"{feedback}\n\nProposed revisions were withheld because validation failed: {issue}"
        return result

    validated_model, validated_data = validated
    if validated_model != original_model:
        result["revised_model"] = validated_model
    if validated_data != original_data:
        result["revised_data"] = validated_data
    return result
