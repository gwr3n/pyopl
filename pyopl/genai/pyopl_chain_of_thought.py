# === Standard library imports ===
import logging
import os
from enum import Enum, auto
from pathlib import Path  # NEW
from typing import (
    Any,
    Callable,  # NEW
    Dict,
    List,  # NEW
    Optional,
    Tuple,  # NEW
    Union,  # NEW
)

# === Local imports ===
from ..pyopl_core import OPLCompiler, SemanticError
from ._strategy_base import (
    GenAIStrategyBase,
)
from ._strategy_base import (
    Grammar as _BaseGrammar,
)
from ._strategy_base import (
    LLMProvider as _BaseLLMProvider,
)
from .genai_pricing import estimate_costs as _estimate_costs  # NEW

# --- Logging Setup ---
# Use module-level logger, and set DEBUG level for development
logger = logging.getLogger(__name__)


# NEW: progress notifier used by generative_solve/feedback and LLM calls
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

# NEW: Few-shot configuration
FEW_SHOT_TOP_K = 3
FEW_SHOT_MAX_CHARS = 2**31 - 1  # soft cap per file to keep prompts manageable

# NEW: Reflexion memory cap
REFLEXION_MAX_MEMORY = 5


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


# NEW: RAG few-shot helpers
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


def extract_json_from_markdown(text: str) -> str:
    return _BASE.extract_json_from_markdown(text)


def _json_loads_relaxed(text: str) -> Any:
    return _BASE.json_loads_relaxed(text)


def _coalesce_response_text(resp) -> str:
    return _BASE._coalesce_response_text(resp)


def _openai_client():
    return _BASE._openai_client()


def _google_client():
    return _BASE._google_client()


def _ollama_generate_text(
    model_name: str, prompt: str, num_predict: Optional[int] = MAX_OUTPUT_TOKENS, return_usage: bool = False
) -> Union[str, Tuple[str, Dict[str, int]]]:  # CHANGED
    """
    Call Ollama's Python client and return the response text.
    If return_usage=True, also return a usage dict with prompt/completion token counts when available.
    """
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
    return _BASE._build_openai_create_params(
        model_name=model_name,
        input_text=input_text,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        expected_json=True,
    )


def _infer_provider(llm_provider: Optional[str], model_name: str) -> LLMProvider:
    base_provider = _BASE.infer_provider(llm_provider, model_name)
    return LLMProvider[base_provider.name]


def _llm_generate_text(
    provider: LLMProvider,
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    progress: Optional[Callable[[str], None]] = None,  # NEW
    capture_usage: bool = False,  # NEW
) -> Union[str, Tuple[str, Dict[str, int]]]:  # CHANGED
    return _BASE.llm_generate_text(
        provider=_BaseLLMProvider[provider.name],
        model_name=model_name,
        input_text=input_text,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        progress=progress,
        capture_usage=capture_usage,
        expected_json=True,
    )


def _call_openai_with_retry(
    client,
    create_params: Dict[str, Any],
    retries: int = 3,
    backoff_sec: float = 1.5,
    progress: Optional[Callable[[str], None]] = None,  # NEW
) -> Any:
    return _BASE._call_openai_with_retry(
        client,
        create_params,
        retries=retries,
        backoff_sec=backoff_sec,
        progress=progress,
    )


# ---------- Prompt builders ----------


def _build_alignment_prompt(prompt: str, grammar_implementation: str, model_code: str, data_code: str) -> str:
    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Judge if the PyOPL model/data fully align with the problem (objective, constraints, variables, indices, and data consistency).\n"
        "Be specific and critical.\n"
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
        "</data>\n"
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
        '- Return ONLY a JSON object with exactly two keys: "aligned" (boolean) and "assessment" (string).\n'
        '- If issues exist, mention the most critical fixes in "assessment", a single short paragraph (3–6 sentences) of plain text.\n'
        "- Do not include any Markdown other than an optional ```json fenced block containing only the JSON.\n"
        "- No bullet lists. No commentary. No additional keys. No trailing commas.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["aligned", "assessment"],\n'
        '  "properties": {\n'
        '    "aligned": {"type": "boolean"},\n'
        '    "assessment": {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        '{ "aligned": false, "assessment": "The model objective function does not include fixed costs." }\n'
        "</example_output>\n"
    )


def _build_cot_generation_prompt(
    prompt: str,
    grammar_implementation: str,
    few_shots: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Chain-of-Thought (Wei et al., 2022) generation prompt.
    Encourages step-by-step reasoning in a hidden scratchpad, while outputting only JSON.
    """
    # Few-shot exemplars (optional)
    few_shots_section = ""
    if few_shots:
        blocks: List[str] = []
        for i, ex in enumerate(few_shots, 1):
            desc_hdr = f'<description path="{ex.get("desc_path", "")}">'
            mod_hdr = f'<model_file path="{ex.get("model_path", "")}">'
            dat_hdr = f'<data_file path="{ex.get("data_path", "")}">'
            blocks.append(
                f'<example index="{i}">\n'
                f"{desc_hdr}\n{ex.get('description','')}\n</description>\n\n"
                f"{mod_hdr}\n{ex.get('model','')}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex.get('data','')}\n</data_file>\n"
                f"</example>\n"
            )
        few_shots_section = (
            "<few_shot_examples>\n"
            "Use these exemplars for structure and syntax inspiration only. Tailor names and indices to the new task.\n"
            + "".join(blocks)
            + "</few_shot_examples>\n\n"
        )

    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Think step by step to derive a correct PyOPL model (.mod) and matching data (.dat) for the problem.\n"
        "First, reason in a private scratchpad to identify sets, parameters, decision variables, objective, and constraints.\n"
        "Ensure indices, domains (binary/integer/float), and data consistency are correct.\n"
        "Then output ONLY the final JSON with the model and data; do not include your scratchpad in the output.\n"
        "If any data are missing, create a small, plausible mock instance consistent with the model.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        f"{few_shots_section}"
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with exactly two keys: "model" and "data".\n'
        "- Each value must be a single JSON string (escape quotes/backslashes, encode newlines as \\n).\n"
        "- Do not include the scratchpad, explanations, or any additional keys.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block that contains only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        '{ "type": "object", "additionalProperties": false,\n'
        '  "required": ["model", "data"],\n'
        '  "properties": { "model": {"type": "string"}, "data": {"type": "string"} } }\n'
        "</json_schema>\n"
        "<hint>\n"
        "Let's think step by step in the scratchpad before writing the final JSON.\n"
        "</hint>\n"
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
    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Answer the user's question about the provided PyOPL model and data.\n"
        "Provide critical, specific feedback. If revisions are necessary for correctness,\n"
        "semantics, or consistency with the grammar reference, propose minimal changes.\n"
        "Only change what is necessary.\n"
        "Label all constraints and the objective function meaningfully; "
        "thoroughly comment the changes to explain the purpose of variables, parameters, objective, and constraints; "
        "match these explanations to user's question by following the predicaments of literate programming.\n"
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


def generative_solve(
    prompt,
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
    few_shot: bool = False,
):
    """Generate a PyOPL model and data file using Chain-of-Thought (CoT) sampling:
    sample -> evaluate (compile + optional alignment), repeated up to `iterations`.

    Signatures and return shape preserved for drop-in compatibility.
    """
    grammar_implementation = _get_grammar_implementation(mode)

    try:
        iterations = max(1, int(iterations))
    except Exception:
        iterations = MAX_ITERATIONS

    do_alignment = ALIGNMENT_CHECK if alignment_check is None else bool(alignment_check)
    provider = _infer_provider(llm_provider, model_name)

    _notify(
        progress,
        f"CoT: provider={provider.value} model={model_name} samples={iterations} alignment={'on' if do_alignment else 'off'}",
    )

    # Few-shot examples (static per run)
    few_shots_list: List[Dict[str, str]] = (
        _gather_few_shots(prompt, k=FEW_SHOT_TOP_K, models_dir=None, progress=progress) if few_shot else []
    )

    assessment_text = ""
    syntax_errors: List[str] = []
    model_code = ""
    data_code = ""

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(iterations):
        # Build CoT generation prompt (no Reflexion memory; independent samples)
        user_prompt = _build_cot_generation_prompt(prompt, grammar_implementation, few_shots=few_shots_list)

        _notify(progress, f"Sample {iteration + 1}/{iterations}: generating (CoT)")
        content, usage = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=user_prompt,
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
        except Exception as e:
            raise RuntimeError(f"Failed to parse model response as JSON: {e}\nResponse: {content}")

        # Compile/evaluate
        compiler = OPLCompiler()
        syntax_errors = []
        try:
            _notify(progress, "Compiling model and data")
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            syntax_errors.append(str(e))
        except Exception as e:
            syntax_errors.append(f"{type(e).__name__}: {e}")

        # Ensure output folder exists and write current attempt
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

        # If compiles and (if enabled) aligns, accept and stop
        if not syntax_errors:
            if do_alignment:
                _notify(progress, "Checking alignment with original prompt...")
                alignment_prompt = _build_alignment_prompt(prompt, grammar_implementation, model_code, data_code)
                alignment_content, usage2 = _llm_generate_text(
                    provider=provider,
                    model_name=model_name,
                    input_text=alignment_prompt,
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
                if not (
                    isinstance(alignment_obj, dict)
                    and isinstance(alignment_obj.get("aligned"), bool)
                    and isinstance(alignment_obj.get("assessment"), str)
                ):
                    raise RuntimeError(f"Invalid alignment response JSON: {alignment_content}")

                assessment_text = alignment_obj.get("assessment", "").strip()
                if alignment_obj["aligned"]:
                    _notify(progress, "Aligned ✓ Stopping.")
                    break
                else:
                    _notify(progress, "Not aligned; trying another CoT sample")
            else:
                _notify(progress, "Compiled ✓ (alignment disabled) Stopping.")
                break
        else:
            _notify(progress, f"Compilation failed with {len(syntax_errors)} error(s); trying another CoT sample")

    # Load the latest attempt from disk
    with open(model_file, "r") as f:
        model_code = f.read()
    with open(data_file, "r") as f:
        data_code = f.read()

    # Final assessment if failed or if alignment disabled
    if syntax_errors or not do_alignment:
        _notify(progress, "Requesting final assessment")
        assessment_prompt = _build_final_assessment_prompt(
            prompt, grammar_implementation, model_code, data_code, syntax_errors
        )
        assessment_text_part, usage4 = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=assessment_prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0 if temperature is not None else None,
            stop=stop,
            progress=progress,
            capture_usage=True,
        )
        total_prompt_tokens += usage4.get("prompt_tokens", 0)
        total_completion_tokens += usage4.get("completion_tokens", 0)
        assessment_text = assessment_text_part or assessment_text

    _notify(progress, "Generation complete")

    # Pricing estimate
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
    prompt,
    model_file,
    data_file,
    model_name=MODEL_NAME,
    mode=Grammar.BNF,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    llm_provider: Optional[str] = LLM_PROVIDER,
    progress: Optional[Callable[[str], None]] = None,  # NEW
):
    """Provide feedback on a given PyOPL model and data file based on a user prompt.

    Args:
        prompt (str): User question or request regarding the model and data.
        model_file (str): Path to the PyOPL model file (.mod).
        data_file (str): Path to the PyOPL data file (.dat).
        model_name (str): LLM model name, e.g. "gpt-5".
        mode (Grammar): Grammar implementation to use: Grammar.NONE, Grammar.BNF, or Grammar.CODE.
        temperature (float|None): Sampling temperature; if None, use model default.
        stop (list[str]|None): List of stop sequences; if None, no stop sequences.
        llm_provider (str|None): "openai" (default), "google", or "ollama".
        progress (callable|None): Optional function that receives progress messages (str).  # NEW

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

    with open(model_file, "r") as fh:
        model_code = fh.read()
    with open(data_file, "r") as fh:
        data_code = fh.read()

    _notify(progress, "Generating feedback from LLM")  # NEW
    user_prompt = _build_feedback_prompt(prompt, grammar_implementation, model_code, data_code)

    content: str = _llm_generate_text(
        provider=provider,
        model_name=model_name,
        input_text=user_prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0 if temperature is not None else None,
        stop=stop,
        progress=progress,  # NEW
        capture_usage=False,
    )
    if not content:
        raise RuntimeError("Empty model response.")
    try:
        _notify(progress, "Feedback received; parsing")  # NEW
        return _json_loads_relaxed(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse feedback response as JSON: {e}\nResponse: {content}")
