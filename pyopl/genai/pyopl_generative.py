# === Standard library imports ===
import json
import logging
import os
import re
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# === Local imports ===
from ..pyopl_core import OPLCompiler, SemanticError
from ._strategy_base import (
    GenAIStrategyBase,
    ImageInput,
    PromptInput,
)
from ._strategy_base import (
    Grammar as _BaseGrammar,
)
from ._strategy_base import (
    LLMProvider as _BaseLLMProvider,
)
from .genai_pricing import estimate_costs as _estimate_costs

# --- Logging Setup ---
# Use module-level logger, and set DEBUG level for development
logger = logging.getLogger(__name__)


# Progress notifier used by generative_solve/feedback and LLM calls
def _notify(progress: Optional[Callable[[str], None]], msg: str) -> None:
    try:
        if progress:
            progress(str(msg))
        else:
            logger.debug(str(msg))
    except Exception:
        # Never let UI callback failures break the run
        pass


MAX_ITERATIONS = 5
MAX_OUTPUT_TOKENS = None
LLM_PROVIDER = "openai"  # "openai", "google", "ollama"
MODEL_NAME = "gpt-5"
ALIGNMENT_CHECK = True  # Whether to check alignment with original prompt

# Few-shot configuration
FEW_SHOT_TOP_K = 3
FEW_SHOT_MAX_CHARS = 2**31 - 1  # soft cap per file to keep prompts manageable

# New: keep image->text short; it's only for retrieval queries
IMAGE_RAG_MAX_TOKENS = 256


_BASE = GenAIStrategyBase(
    logger=logger,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    few_shot_top_k=FEW_SHOT_TOP_K,
    few_shot_max_chars=FEW_SHOT_MAX_CHARS,
)


class LLMProvider(Enum):
    OPENAI = "openai"  # Default
    GOOGLE = "google"
    OLLAMA = "ollama"


class Grammar(Enum):
    NONE = auto()
    BNF = auto()
    CODE = auto()


# ---------- Utilities ----------


def _normalize_prompt_input(prompt: PromptInput) -> Tuple[str, List[ImageInput]]:
    """
    Normalize the public `prompt` argument into (prompt_text, prompt_images).

    Supported:
      - str
      - {"text": "...", "images": [ImageInput|str|Path|dict]}
      - {"text": "...", "image": ImageInput|str|Path|dict}
    """
    return _BASE.normalize_prompt_input(prompt)


def _read_file(path: str) -> str:
    return _BASE.read_file(path)


def _read_pyopl_GBNF() -> str:
    return _BASE.read_pyopl_GBNF()


def _read_pyopl_grammar() -> str:
    return _BASE.read_pyopl_grammar()


def _read_pyopl_code() -> str:
    return _BASE.read_pyopl_code()


def _get_grammar_implementation(mode: Grammar) -> str:
    return _BASE.get_grammar_implementation(_BaseGrammar[mode.name])


# RAG few-shot helpers
def _safe_read_text(path: Path, max_chars: int = FEW_SHOT_MAX_CHARS) -> str:
    return _BASE.safe_read_text(path, max_chars=max_chars)


def _find_pair_in_folder(desc_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Given a description .txt path, locate associated .mod and .dat in the same folder.
    Preference order:
      1) Same stem: <stem>.mod and <stem>.dat
      2) First *.mod and first *.dat in folder (sorted)
    """
    return _BASE.find_pair_in_folder(desc_path)


def _gather_few_shots(
    problem_description: str,
    k: int = FEW_SHOT_TOP_K,
    models_dir: Optional[str | Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, str]]:
    """
    Use rag_helper to find top-k relevant examples and return a list of dicts with keys:
      - description (str)
      - model (str)
      - data (str)
      - desc_path / model_path / data_path (optional metadata)
    """
    return _BASE.gather_few_shots(problem_description, k=k, models_dir=models_dir, progress=progress)


# Central renderer for few-shot exemplars (to avoid duplication)
def _render_few_shots_section(few_shots: Optional[List[Dict[str, str]]]) -> str:
    return _BASE.render_few_shots_section(few_shots)


def extract_json_from_markdown(text: str) -> str:
    """
    Extract JSON object from a Markdown code block if present.
    Fallback: find the first balanced {...} JSON object.
    """
    # First try fenced block
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: balanced brace scan
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text


def _json_loads_relaxed(text: str) -> Dict[str, Any]:
    """
    Try to parse as JSON; if it fails, attempt fenced JSON extraction first.
    """
    try:
        return json.loads(text)
    except Exception:
        return json.loads(extract_json_from_markdown(text))


def _coalesce_response_text(resp) -> str:
    # Prefer SDK convenience if present
    if getattr(resp, "output_text", None):
        return resp.output_text or ""

    # Responses API: try common shapes
    try:
        chunks = []
        for item in getattr(resp, "output", []) or []:
            content_blocks = getattr(item, "content", None)
            if content_blocks is None:
                if hasattr(item, "text"):
                    chunks.append(getattr(item, "text") or "")
                continue
            for block in content_blocks:
                if hasattr(block, "text"):
                    chunks.append(getattr(block, "text") or "")
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
        if chunks:
            return "".join(chunks)
    except Exception:
        pass

    # Last-resort fallbacks
    try:
        first = getattr(resp, "output", [])[0]
        first_content = getattr(first, "content", [])[0]
        if hasattr(first_content, "text"):
            return first_content.text or ""
    except Exception:
        pass
    return ""


def _openai_client():
    return _BASE._openai_client()


def _google_client():
    return _BASE._google_client()


def _ollama_generate_text(
    model_name: str, prompt: str, num_predict: Optional[int] = MAX_OUTPUT_TOKENS, return_usage: bool = False
) -> Union[str, Tuple[str, Dict[str, int]]]:
    """
    Call Ollama's Python client and return the response text.
    If return_usage=True, also return a usage dict with prompt/completion token counts when available.
    """
    # Generative strategy expects JSON-structured outputs; enforce JSON format when possible.
    return _BASE._ollama_generate_text(
        model_name=model_name,
        prompt=prompt,
        num_predict=num_predict,
        return_usage=return_usage,
        enforce_json=True,
    )


def _build_create_params(
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> Dict[str, Any]:
    # Preserve prior behavior: always request JSON from OpenAI when possible.
    return _BASE._build_openai_create_params(
        model_name=model_name,
        input_text=input_text,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        expected_json=True,
    )


def _infer_provider(llm_provider: Optional[str], model_name: str) -> LLMProvider:
    base_p = _BASE.infer_provider(llm_provider, model_name)
    return LLMProvider(base_p.value)


def _llm_generate_text(
    provider: LLMProvider,
    model_name: str,
    input_text: str,
    *,
    images: Optional[List[ImageInput]] = None,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
    capture_usage: bool = False,
    expected_json: bool = True,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    # Preserve prior behavior by default: JSON when possible (generation/alignment/revision/feedback)
    base_provider = _BaseLLMProvider(provider.value)
    return _BASE.llm_generate_text(
        provider=base_provider,
        model_name=model_name,
        input_text=input_text,
        images=images,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        progress=progress,
        capture_usage=capture_usage,
        expected_json=expected_json,
    )


def _describe_images_for_rag(
    *,
    provider: LLMProvider,
    model_name: str,
    images: List[ImageInput],
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Produce a compact textual description of images to improve BERT-based few-shot retrieval.
    Best-effort: failures return empty string.
    """
    if not images:
        return ""

    # Current strategy base does not support images for Ollama.
    if provider == LLMProvider.OLLAMA:
        _notify(progress, "[RAG] Image context skipped (Ollama image prompts not supported)")
        return ""

    prompt = (
        "You will be shown one or more images that may contain an optimization problem statement.\n"
        "Write a concise textual description to help retrieve similar example problems.\n"
        "Priorities:\n"
        "1) Transcribe any readable text verbatim (especially numbers, units, table headers/rows).\n"
        "2) If tables exist, summarize their structure and key entries.\n"
        "3) If charts/diagrams exist, describe labels, axes, and relationships.\n"
        "Output plain text only. No JSON. No Markdown.\n"
    )

    try:
        _notify(progress, "[RAG] Generating image context for few-shot retrieval")
        text = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=prompt,
            images=images,
            max_tokens=IMAGE_RAG_MAX_TOKENS,
            temperature=0.0,
            stop=None,
            progress=progress,
            capture_usage=False,
            expected_json=False,
        )
        if isinstance(text, tuple):
            text = text[0]
        return (text or "").strip()
    except Exception as e:
        logger.debug(f"Image->text for RAG failed: {e}")
        _notify(progress, "[RAG] Image context generation failed; continuing with text-only retrieval")
        return ""


# ---------- Prompt builders ----------


# Shared commenting guidance for prompts
def _commenting_guidelines() -> str:
    return (
        "Label the objective and each constraint. "
        "Add concise comments explaining variables, parameters, and constraints, "
        "aligned to the problem (literate style).\n"
    )


def _revision_guidelines_syntax() -> str:
    return (
        "<revision_guidelines>\n"
        "- Fix the listed syntax/semantic errors.\n"
        "- Make the minimal set of changes necessary to correct syntax/semantic errors.\n"
        "- Preserve the original modeling structure when possible.\n"
        "- Ensure the objective, constraints, indices, and variable domains reflect the problem description.\n"
        "- Keep syntax strictly valid.\n"
        "- Return complete model and data strings; do not return diffs.\n"
        "</revision_guidelines>\n\n"
    )


def _revision_guidelines_alignment() -> str:
    return (
        "<revision_guidelines>\n"
        "- Address the alignment issues noted in the assessment.\n"
        "- Make the minimal set of changes necessary to correct misalignment.\n"
        "- Preserve the original modeling structure when possible.\n"
        "- Ensure the objective, constraints, indices, and variable domains reflect the problem description.\n"
        "- Keep syntax strictly valid.\n"
        "- Return complete model and data strings; do not return diffs.\n"
        "</revision_guidelines>\n\n"
    )


def _build_generation_prompt(
    prompt: PromptInput, grammar_implementation: str, few_shots: Optional[List[Dict[str, str]]] = None
) -> str:
    few_shots_section = _render_few_shots_section(few_shots)
    commenting_guidelines = _commenting_guidelines()

    return (
        "<role>\nYou are an expert in mathematical optimization and PyOPL.\n</role>\n\n"
        "<task>\n"
        "Think step by step to produce a syntactically valid PyOPL model (.mod) and matching data (.dat) that faithfully implement the problem.\n"
        "First, reason in a private scratchpad to identify sets, parameters, decision variables, objective, and constraints.\n"
        "Ensure indices, domains (binary/integer/float), and data are correct and consistent with the problem description.\n"
        "Choose correct domains (binary/integer/float) from context. Add clear labels and explanatory comments.\n"
        f"{commenting_guidelines}"
        "If any data are missing, create a small, plausible mock instance consistent with the model.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "</task>\n\n"
        "<grammar_reference>\n--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n</grammar_reference>\n\n"
        f"{few_shots_section}"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<output_requirements>\n"
        "- Output ONLY the final JSON with the model and data; do not include your scratchpad in the output.\n"
        '- Return ONLY a JSON object with keys "model" and "data". Values are single strings; escape quotes and backslashes; encode newlines as \\n. No extra keys.\n'
        "- You MAY wrap the JSON in a ```json fence containing only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        '{ "type": "object", "additionalProperties": false, "required": ["model","data"], '
        '"properties": { "model":{"type":"string"}, "data":{"type":"string"} } }\n'
        "</json_schema>\n"
        "<example_output>\n"
        "{\n"
        '  "model": "// minimal example\\nfloat a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to {\\n  c1: b*x >= 0;\\n}\\n",\n'
        '  "data":  "a = 10;\\n b = 5;"\n'
        "}\n"
        "</example_output>\n"
    )


def _build_alignment_prompt(prompt: str, grammar_implementation: str, model_code: str, data_code: str) -> str:
    return (
        "<role>\nYou are an expert in mathematical optimization and PyOPL.\n</role>\n\n"
        "<task>\n"
        "Judge if the PyOPL model/data fully align with the problem (objective, constraints, variables, indices, and data consistency).\n"
        "Be specific and critical.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "</task>\n\n"
        "<grammar_reference>\n--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n</grammar_reference>\n\n"
        "<inputs>\n"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n"
        "</inputs>\n\n"
        "<assessment_focus>\n"
        "- Objective and constraints reflect the prompt intent.\n"
        "- Decision variables have correct domains and indices.\n"
        "- Data is consistent with sets/parameters used by the model.\n"
        "- Signs, units, and indexing are correct; no missing links.\n"
        "- Any critical omissions or extraneous constraints.\n"
        "- Most impactful improvements if misaligned.\n"
        "</assessment_focus>\n\n"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with exactly two keys: "aligned" (boolean) and "assessment" (string).\n'
        '- If issues exist, mention the most critical fixes in "assessment", a single short paragraph (3–6 sentences) of plain text.\n'
        "- Do not include any Markdown other than an optional ```json fenced block containing only the JSON.\n"
        "- No bullet lists. No commentary. No additional keys. No trailing commas.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        '{ "type":"object", "additionalProperties": false, "required":["aligned","assessment"], '
        '"properties": { "aligned":{"type":"boolean"}, "assessment":{"type":"string"} } }\n'
        "</json_schema>\n"
        "<example_output>\n"
        '{ "aligned": false, "assessment": "The capacity constraint omits fixed setup costs and the data set D is unused." }\n'
        "</example_output>\n"
    )


# Unified revision prompt for both syntax errors and alignment issues
def _build_revision_prompt(
    prompt: str,
    grammar_implementation: str,
    model_code: str,
    data_code: str,
    compile_errors: Optional[List[str]] = None,
    alignment_assessment: Optional[str] = None,
    few_shots: Optional[List[Dict[str, str]]] = None,
) -> str:
    few_shots_section = _render_few_shots_section(few_shots)
    commenting_guidelines = _commenting_guidelines()

    errors_block = ""
    if compile_errors:
        joined = "\n".join(f"- {e}" for e in compile_errors)
        errors_block = f"<errors>\n{joined}\n</errors>\n\n"

    assess_block = ""
    if alignment_assessment:
        assess_block = f"<alignment_assessment>\n{alignment_assessment}\n</alignment_assessment>\n\n"

    revision_guidelines = (
        _revision_guidelines_syntax() if compile_errors and len(compile_errors) > 0 else _revision_guidelines_alignment()
    )

    return (
        "<role>\nYou are an expert in mathematical optimization and PyOPL.\n</role>\n\n"
        "<task>\n"
        "Revise the model/data to resolve the specified issues while preserving the intended formulation.\n"
        "Change only what is necessary; keep syntax valid.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        f"{commenting_guidelines}"
        "Use the PyOPL reference strictly for syntax.\n"
        "</task>\n\n"
        f"{revision_guidelines}"
        "<grammar_reference>\n--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n</grammar_reference>\n\n"
        f"{few_shots_section}"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<previous_attempt>\n<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n</previous_attempt>\n\n"
        f"{errors_block}"
        f"{assess_block}"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with keys "model" and "data". Values are single strings; escape quotes/backslashes; encode newlines as \\n.\n'
        "- You MAY wrap the JSON in a ```json fence containing only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        '{ "type":"object", "additionalProperties": false, "required":["model","data"], '
        '"properties": { "model":{"type":"string"}, "data":{"type":"string"} } }\n'
        "</json_schema>\n"
        "<example_output>\n"
        "{\n"
        '  "model": "// revised example\\nfloat a;\\nfloat b;\\ndvar float x >= 0;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
        '  "data":  "a = 10;\\nb= 5;"\n'
        "}\n"
        "</example_output>\n"
    )


def _build_final_assessment_prompt(
    prompt: str, grammar_implementation: str, model_code: str, data_code: str, syntax_errors
) -> str:
    syntax_errors_str = f"SYNTAX ERRORS:\n{syntax_errors}\n\n" if syntax_errors else ""
    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Judge if the PyOPL model/data fully align with the problem (objective, constraints, variables, indices, and data consistency).\n"
        "Be specific and critical.\n"
        "If you believe the problem description is incomplete or ambiguous, point this out in your assessment.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        "<inputs>\n"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n\n"
        f"{syntax_errors_str}"
        "</inputs>\n\n"
        "<assessment_focus>\n"
        "- Objective and constraints reflect the prompt intent.\n"
        "- Decision variables have correct domains and indices.\n"
        "- Data is consistent with sets/parameters used by the model.\n"
        "- Signs, units, and indexing are correct; no missing links.\n"
        "- Any syntax error raised by the compiler.\n"
        "- Most impactful improvements if misaligned.\n"
        "</assessment_focus>\n\n"
        "<output_requirements>\n"
        "- Return a single short paragraph (3–6 sentences) of plain text.\n"
        "- No Markdown, no bullet lists, no code fences.\n"
        "- If issues exist, mention the most critical fixes.\n"
        "</output_requirements>\n"
    )


def _build_feedback_prompt(user_prompt_text: str, grammar_implementation: str, model_code: str, data_code: str) -> str:
    guidelines = (
        "Label the objective and each constraint. "
        "Include concise comments explaining variables, parameters, and constraints, "
        "aligned to the user's question and the problem (literate style).\n"
    )

    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Answer the user's question about the provided PyOPL model and data.\n"
        "Provide critical, specific feedback. If revisions are necessary for correctness,\n"
        "semantics, or consistency with the grammar reference, propose minimal changes.\n"
        "Only change what is necessary.\n"
        f"{guidelines}"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        "<inputs>\n"
        "<question>\n"
        f"{user_prompt_text}\n"
        "</question>\n\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n"
        "</inputs>\n\n"
        "<output_requirements>\n"
        "- Return ONLY a JSON object with 1 required key and up to 2 optional keys:\n"
        '  "feedback" (required), "revised_model" (optional), "revised_data" (optional).\n'
        "- Each value must be a single JSON string. Escape all double quotes and backslashes;\n"
        "  encode newlines as \\n.\n"
        '- If no changes are needed, omit "revised_model" and "revised_data".\n'
        "- If changes are needed, return complete model and data strings; do not return diffs.\n"
        "- No trailing commas. No additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["feedback"],\n'
        '  "properties": {\n'
        '    "feedback": {"type": "string"},\n'
        '    "revised_model": {"type": "string"},\n'
        '    "revised_data": {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        "{\n"
        '  "feedback": "The model was missing coefficients a and b.",\n'
        '  "revised_model": "// minimal fix\\nfloat a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
        '  "revised_data":  "a = 10;\\nb= 5;"\n'
        "}\n"
        "</example_output>\n"
    )


# ---------- Public API ----------

try:
    from .pyopl_generative_graphchain import generative_solve_graphchain
except ImportError:
    # Fallback if graphchain module not yet available
    generative_solve_graphchain = None


def generative_solve(
    prompt: PromptInput,
    model_file,
    data_file,
    model_name=MODEL_NAME,
    mode=Grammar.BNF,
    iterations=MAX_ITERATIONS,
    return_statistics=False,
    alignment_check: Optional[bool] = None,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    llm_provider: Optional[str] = LLM_PROVIDER,
    progress: Optional[Callable[[str], None]] = None,
    few_shot: bool = True,
    use_graphchain: bool = True,
):
    """Generate a PyOPL model and data file from a prompt, validate with pyopl, iterate on errors, and assess alignment.

    Args:
        prompt (PromptInput): The problem description, as a string or dict with "text" and optional "images".
        model_file (str): Path to save the generated PyOPL model (.mod).
        data_file (str): Path to save the generated PyOPL data file (.dat).
        model_name (str): LLM model name, e.g. "gpt-5".
        mode (Grammar): Grammar implementation to use: Grammar.NONE, Grammar.BNF, or Grammar.CODE.
        iterations (int): Maximum number of generation/validation iterations (default 5).
        return_statistics (bool): If True, return a dict with statistics instead of just the assessment string.
        alignment_check (bool|None): If True, check alignment with the original prompt; if False, skip alignment check; if None, use default ALIGNMENT_CHECK.
        temperature (float|None): Sampling temperature; if None, use model default.
        stop (list[str]|None): List of stop sequences; if None, no stop sequences.
        llm_provider (str|None): "openai" (default), "google", or "ollama".
        progress (callable|None): Optional function that receives progress messages (str).

    Returns:
        str or dict: If return_statistics is False, returns the final assessment string.
                     If return_statistics is True, returns a dict with keys:
                     - "iterations": number of iterations performed
                     - "assessment": final assessment string
                     - "syntax_errors": list of syntax errors encountered (if any)
                     - "cost": { "model": str, "usage": {"prompt_tokens": int, "completion_tokens": int}, "estimated_costs": dict }
    Raises:
        RuntimeError: If generation or validation fails irrecoverably.
    """
    if use_graphchain and generative_solve_graphchain is not None:
        return generative_solve_graphchain(
            prompt=prompt,
            model_file=model_file,
            data_file=data_file,
            model_name=model_name,
            mode=mode,
            iterations=iterations,
            return_statistics=return_statistics,
            alignment_check=alignment_check,
            temperature=temperature,
            stop=stop,
            llm_provider=llm_provider,
            progress=progress,
            few_shot=few_shot,
        )

    prompt_text, prompt_images = _normalize_prompt_input(prompt)

    # Progress notifier used by generative_solve/feedback and LLM calls
    def _notify(progress: Optional[Callable[[str], None]], msg: str) -> None:
        try:
            if progress:
                progress(str(msg))
            else:
                logger.debug(str(msg))
        except Exception:
            # Never let UI callback failures break the run
            pass

    _notify(
        progress,
        f"Generating with provider={_infer_provider(llm_provider, model_name).value} model={model_name} iterations={iterations} alignment={'on' if (ALIGNMENT_CHECK if alignment_check is None else bool(alignment_check)) else 'off'}",
    )

    grammar_implementation = _get_grammar_implementation(mode)

    try:
        iterations = max(1, int(iterations))
    except Exception:
        iterations = MAX_ITERATIONS

    do_alignment = ALIGNMENT_CHECK if alignment_check is None else bool(alignment_check)
    provider = _infer_provider(llm_provider, model_name)

    _notify(
        progress,
        f"Generating with provider={provider.value} model={model_name} iterations={iterations} alignment={'on' if do_alignment else 'off'}",
    )

    # Retrieve few-shot examples using RAG:
    # - Base query: prompt text
    # - If images exist: append a compact image-derived textual context
    rag_query_text = prompt_text
    if prompt_images:
        img_context = _describe_images_for_rag(
            provider=provider,
            model_name=model_name,
            images=prompt_images,
            progress=progress,
        )
        if img_context:
            _notify(progress, "[RAG] Image context: " + img_context[:80] + ("..." if len(img_context) > 80 else ""))
            rag_query_text = f"{prompt_text}\n\n[IMAGE_CONTEXT]\n{img_context}\n"

    few_shots: List[Dict[str, str]] = (
        _gather_few_shots(rag_query_text, k=FEW_SHOT_TOP_K, models_dir=None, progress=progress) if few_shot else []
    )

    user_prompt = _build_generation_prompt(prompt_text, grammar_implementation, few_shots=few_shots)
    assessment_text = ""
    syntax_errors: list[str] = []

    # Aggregate token usage across all LLM calls in this run
    total_prompt_tokens = 0
    total_completion_tokens = 0

    model_code = ""
    data_code = ""

    for iteration in range(iterations):
        logger.debug(f"Iteration {iteration + 1}/{iterations}")
        _notify(progress, f"Iteration {iteration + 1}/{iterations}: prompting model")

        content, usage = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=user_prompt,
            images=prompt_images,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            stop=stop,
            progress=progress,
            capture_usage=True,
        )
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        if not content:
            raise RuntimeError("Empty model response.")
        try:
            result = _json_loads_relaxed(content)
            model_code = result["model"]
            data_code = result["data"]
            _notify(progress, "LLM response parsed (model + data)")
            logger.debug("Model and data generated.")
        except Exception as e:
            raise RuntimeError(f"Failed to parse model response as JSON: {e}\nResponse: {content}")

        compiler = OPLCompiler()
        syntax_errors = []
        try:
            _notify(progress, "Compiling model and data")
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            syntax_errors.append(str(e))
            logger.debug(f"Semantic error in model: {e}")
        except Exception as e:
            syntax_errors.append(f"{type(e).__name__}: {e}")

        # Ensure output folder exists and write files
        model_dir = os.path.dirname(model_file)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        data_dir = os.path.dirname(data_file)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        with open(model_file, "w") as f:
            f.write(model_code)
        with open(data_file, "w") as f:
            f.write(data_code)
        _notify(progress, f"Wrote files: {model_file} • {data_file}")

        if not syntax_errors:
            if not do_alignment:
                _notify(progress, "Syntax OK; alignment check disabled. Stopping.")
                break

            logger.debug("Checking alignment with original prompt...")
            _notify(progress, "Checking alignment with original prompt...")
            alignment_prompt = _build_alignment_prompt(prompt_text, grammar_implementation, model_code, data_code)

            alignment_content, usage2 = _llm_generate_text(
                provider=provider,
                model_name=model_name,
                input_text=alignment_prompt,
                images=prompt_images,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0 if temperature is not None else None,
                stop=stop,
                progress=progress,
                capture_usage=True,
            )
            total_prompt_tokens += usage2.get("prompt_tokens", 0)
            total_completion_tokens += usage2.get("completion_tokens", 0)

            if not alignment_content:
                raise RuntimeError("Empty alignment response.")

            alignment_obj = _json_loads_relaxed(alignment_content)
            if (
                isinstance(alignment_obj, dict)
                and isinstance(alignment_obj.get("aligned"), bool)
                and isinstance(alignment_obj.get("assessment"), str)
            ):
                assessment_text = alignment_obj.get("assessment", "").strip()
                if alignment_obj["aligned"]:
                    _notify(progress, "Aligned ✓ Stopping.")
                    logger.debug("Model and data are syntactically valid and aligned with the prompt.")
                    break
                else:
                    _notify(progress, "Not aligned; revising per assessment")
                    logger.debug(
                        f"Model and data are syntactically valid but NOT aligned with the prompt. Assessment: {assessment_text}"
                    )
                    user_prompt = _build_revision_prompt(
                        prompt=prompt_text,
                        grammar_implementation=grammar_implementation,
                        model_code=model_code,
                        data_code=data_code,
                        compile_errors=None,
                        alignment_assessment=assessment_text,
                        few_shots=few_shots,
                    )
            else:
                raise RuntimeError(f"Invalid alignment response JSON: {alignment_content}")
        else:
            _notify(progress, f"Syntax/semantic errors found: {len(syntax_errors)}; revising...")
            logger.debug("Model or data has syntax errors; revising...")
            user_prompt = _build_revision_prompt(
                prompt=prompt_text,
                grammar_implementation=grammar_implementation,
                model_code=model_code,
                data_code=data_code,
                compile_errors=syntax_errors,
                alignment_assessment=None,
                few_shots=few_shots,
            )

    # Load latest version of the model and data files
    with open(model_file, "r") as f:
        model_code = f.read()
    with open(data_file, "r") as f:
        data_code = f.read()

    if syntax_errors or not do_alignment:
        logger.debug("Final assessment of model and data alignment...")
        _notify(progress, "Requesting final assessment")
        assessment_prompt = _build_final_assessment_prompt(
            prompt_text, grammar_implementation, model_code, data_code, syntax_errors
        )
        # Capture usage and unpack directly for mypy
        assessment_text_part, usage3 = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=assessment_prompt,
            images=prompt_images,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0 if temperature is not None else None,
            stop=stop,
            progress=progress,
            capture_usage=True,
        )
        total_prompt_tokens += usage3.get("prompt_tokens", 0)
        total_completion_tokens += usage3.get("completion_tokens", 0)
        assessment_text = assessment_text_part or ""

    _notify(progress, "Generation complete")

    # Pricing estimate using aggregated usage
    try:
        from types import SimpleNamespace
    except Exception:
        SimpleNamespace = None  # type: ignore

    usage_summary = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }
    estimated_costs: Dict[str, Any] = {}
    if SimpleNamespace is not None:
        try:
            args = SimpleNamespace(model=model_name)
            estimated_costs = _estimate_costs(args, usage_summary) or {}
        except Exception:
            estimated_costs = {}
    cost = {
        "model": model_name,
        "usage": usage_summary,
        "estimated_costs": estimated_costs,
    }
    _notify(progress, f"[LLM] Estimated costs: {cost}")

    if return_statistics:
        return {
            "iterations": iteration + 1,
            "assessment": assessment_text.strip(),
            "syntax_errors": syntax_errors,
            "cost": cost,
        }
    else:
        return assessment_text.strip()


def generative_feedback(
    prompt: PromptInput,
    model_file,
    data_file,
    model_name=MODEL_NAME,
    mode=Grammar.BNF,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    llm_provider: Optional[str] = LLM_PROVIDER,
    progress: Optional[Callable[[str], None]] = None,
):
    """Provide feedback on a given PyOPL model and data file based on a user prompt.

    Args:
        prompt (PromptInput): User question or request for feedback about the model/data.
        model_file (str): Path to the PyOPL model file (.mod).
        data_file (str): Path to the PyOPL data file (.dat).
        model_name (str): LLM model name, e.g. "gpt-5".
        mode (Grammar): Grammar implementation to use: Grammar.NONE, Grammar.BNF, or Grammar.CODE.
        temperature (float|None): Sampling temperature; if None, use model default.
        stop (list[str]|None): List of stop sequences; if None, no stop sequences.
        llm_provider (str|None): "openai" (default), "google", or "ollama".
        progress (callable|None): Optional function that receives progress messages (str).

    Raises:
        RuntimeError: If feedback generation fails irrecoverably.

    Returns:
        dict: A dictionary with keys:
              - "feedback": string with the feedback message
              - "revised_model": (optional) string with revised PyOPL model if changes are proposed
              - "revised_data": (optional) string with revised PyOPL data if changes are proposed
    """
    provider = _infer_provider(llm_provider, model_name)
    grammar_implementation = _get_grammar_implementation(mode)

    prompt_text, prompt_images = _normalize_prompt_input(prompt)

    with open(model_file, "r") as fh:
        model_code = fh.read()
    with open(data_file, "r") as fh:
        data_code = fh.read()

    _notify(progress, "Generating feedback from LLM")
    user_prompt = _build_feedback_prompt(prompt_text, grammar_implementation, model_code, data_code)

    content: str = _llm_generate_text(
        provider=provider,
        model_name=model_name,
        input_text=user_prompt,
        images=prompt_images,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0 if temperature is not None else None,
        stop=stop,
        progress=progress,
        capture_usage=False,
    )
    if not content:
        raise RuntimeError("Empty model response.")
    try:
        _notify(progress, "Feedback received; parsing")
        parsed = _json_loads_relaxed(content)

        # Normalize common string-escaping issues: some LLMs return JSON
        # where newline/tab sequences are double-escaped (literal "\\n").
        # If we detect that pattern (escaped sequences present but no real
        # newlines), attempt a safe unescape for the common fields so the
        # UI receives readable text.
        def _maybe_unescape(s: Any) -> Any:
            if not isinstance(s, str):
                return s
            # Only attempt when we see escaped sequences but no real ones
            if "\\n" in s and "\n" not in s:
                try:
                    return s.encode("utf-8").decode("unicode_escape")
                except Exception:
                    return s.replace("\\n", "\n").replace("\\t", "\t")
            return s

        if isinstance(parsed, dict):
            for key in ("feedback", "revised_model", "revised_data", "assessment", "message"):
                if key in parsed:
                    parsed[key] = _maybe_unescape(parsed[key])
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to parse feedback response as JSON: {e}\nResponse: {content}")
