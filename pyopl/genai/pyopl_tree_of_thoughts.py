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

# Reflexion memory cap
REFLEXION_MAX_MEMORY = 5

# Tree-of-Thoughts search parameters
TOT_BRANCH_FACTOR = 3  # number of candidate thoughts per node (K)
TOT_BEAM_WIDTH = 3  # number of best thoughts kept per level (B)


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


def extract_json_from_markdown(text: str) -> str:
    """
    Extract a top-level JSON value (array or object) from a Markdown code block if present.
    Fallback: find the first balanced {...} or [...] JSON value (quote-aware).
    """
    # 1) Try fenced block (```json ... ``` or ``` ... ```)
    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    # 2) Find first top-level { or [ and return balanced segment
    opener = None
    start = -1
    for i, ch in enumerate(text):
        if ch == "{" or ch == "[":
            opener = ch
            start = i
            break
    if start == -1:
        return text

    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def _json_loads_relaxed(text: str) -> Any:
    """
    Parse JSON from raw text, allowing for Markdown fences and surrounding prose.
    Tries multiple candidates and returns the first parsed JSON object/array.
    """
    # 0) Try raw first
    try:
        return json.loads(text)
    except Exception:
        pass

    def _balanced_slice(s: str, start: int, opener: str, closer: str) -> Optional[str]:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(s)):
            c = s[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    candidates: list[str] = []

    # 1) All fenced code blocks (prefer ```json)
    fenced = list(re.finditer(r"```(\w+)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE))
    json_pref: list[str] = []
    other_pref: list[str] = []
    for m in fenced:
        lang = (m.group(1) or "").lower().strip()
        body = (m.group(2) or "").strip()
        if body:
            (json_pref if lang == "json" else other_pref).append(body)
    candidates.extend(json_pref + other_pref)

    # 2) Every balanced JSON array starting at each '['
    seen = set()
    for i, ch in enumerate(text):
        if ch == "[":
            frag = _balanced_slice(text, i, "[", "]")
            if frag and frag not in seen:
                seen.add(frag)
                candidates.append(frag)

    # 3) Every balanced JSON object starting at each '{'
    for i, ch in enumerate(text):
        if ch == "{":
            frag = _balanced_slice(text, i, "{", "}")
            if frag and frag not in seen:
                seen.add(frag)
                candidates.append(frag)

    # 4) Legacy single-extract heuristic
    try:
        legacy = extract_json_from_markdown(text).strip()
        if legacy and legacy not in seen:
            candidates.append(legacy)
    except Exception:
        pass

    # Prefer arrays of objects with "model" and "data"
    def _score(obj: Any) -> int:
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            has_keys = sum(1 for x in obj if "model" in x and "data" in x)
            return 100 + has_keys
        if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
            inner = obj["candidates"]
            has_keys = sum(1 for x in inner if isinstance(x, dict) and "model" in x and "data" in x)
            return 90 + has_keys
        if isinstance(obj, list):
            return 50
        if isinstance(obj, dict):
            return 40
        return 0

    best_obj = None
    best_score = -1
    first_exception: Optional[Exception] = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            sc = _score(obj)
            if sc > best_score:
                best_score = sc
                best_obj = obj
                if sc >= 100:
                    return obj
        except Exception as e:
            if first_exception is None:
                first_exception = e

    if best_obj is not None:
        return best_obj

    # Last resort
    try:
        return json.loads((text or "").strip())
    except Exception as e:
        raise e if first_exception is None else first_exception


def _coalesce_response_text(resp) -> str:
    return _BASE._coalesce_response_text(resp)


def _openai_client():
    return _BASE._openai_client()


def _google_client():
    return _BASE._google_client()


def _ollama_generate_text(
    model_name: str,
    prompt: str,
    num_predict: Optional[int] = MAX_OUTPUT_TOKENS,
    return_usage: bool = False,
    enforce_json: bool = False,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    """
    Call Ollama's Python client and return the response text.
    If return_usage=True, also return a usage dict with prompt/completion token counts when available.
    """
    return _BASE._ollama_generate_text(
        model_name=model_name,
        prompt=prompt,
        num_predict=num_predict,
        return_usage=return_usage,
        enforce_json=enforce_json,
    )


def _build_create_params(
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    expected_json: bool = False,
) -> Dict[str, Any]:
    return _BASE._build_openai_create_params(
        model_name=model_name,
        input_text=input_text,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        expected_json=expected_json,
    )


def _infer_provider(llm_provider: Optional[str], model_name: str) -> LLMProvider:
    return LLMProvider[_BASE.infer_provider(llm_provider, model_name).name]


def _llm_generate_text(
    provider: LLMProvider,
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    progress: Optional[Callable[[str], None]] = None,
    capture_usage: bool = False,
    expected_json: bool = False,
) -> Union[str, Tuple[str, Dict[str, int]]]:
    return _BASE.llm_generate_text(
        provider=_BaseLLMProvider[provider.name],
        model_name=model_name,
        input_text=input_text,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        progress=progress,
        capture_usage=capture_usage,
        expected_json=expected_json,
    )


def _call_openai_with_retry(
    client,
    create_params: Dict[str, Any],
    retries: int = 3,
    backoff_sec: float = 1.5,
    progress: Optional[Callable[[str], None]] = None,
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
        "- Each value must be a valid JSON string containing the full text. Use standard JSON escaping only.\n"
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
    """Generate a PyOPL model and data file using Tree-of-Thoughts (ToT):
    breadth-bounded expansion -> evaluate (compile + optional alignment) -> beam select, up to `iterations` depth.

    Signatures and return shape preserved for drop-in compatibility.
    """
    from dataclasses import dataclass  # (scoped import to minimize global churn)

    grammar_implementation = _get_grammar_implementation(mode)

    try:
        depth_limit = max(1, int(iterations))
    except Exception:
        depth_limit = MAX_ITERATIONS

    do_alignment = ALIGNMENT_CHECK if alignment_check is None else bool(alignment_check)
    provider = _infer_provider(llm_provider, model_name)

    # Few-shot examples (static per run)
    few_shots_list: List[Dict[str, str]] = (
        _gather_few_shots(prompt, k=FEW_SHOT_TOP_K, models_dir=None, progress=progress) if few_shot else []
    )

    _notify(
        progress,
        f"ToT: provider={provider.value} model={model_name} depth={depth_limit} K={TOT_BRANCH_FACTOR} B={TOT_BEAM_WIDTH} alignment={'on' if do_alignment else 'off'}",
    )

    @dataclass
    class ThoughtNode:
        model_code: str
        data_code: str
        score: float
        aligned: bool
        assessment: str
        syntax_errors: List[str]
        level: int

    def _score_candidate(model_code: str, data_code: str) -> Tuple[float, bool, str, List[str], Dict[str, int]]:
        """Compile and optionally alignment-check a candidate. Returns (score, aligned, assessment, errors, usage)."""
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        compiler = OPLCompiler()
        errors: List[str] = []
        try:
            _notify(progress, "Compiling candidate")
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")

        compiled_ok = len(errors) == 0
        aligned = False
        assessment = ""

        # Base score: emphasize compilation success
        score = 2.0 if compiled_ok else 0.0
        # If not compiled, penalize by rough error length to break ties
        if not compiled_ok:
            score -= min(1.0, 0.001 * sum(len(err) for err in errors))

        if compiled_ok and do_alignment:
            _notify(progress, "Checking alignment for candidate...")
            alignment_prompt = _build_alignment_prompt(prompt, grammar_implementation, model_code, data_code)
            alignment_content, usage = _llm_generate_text(
                provider=provider,
                model_name=model_name,
                input_text=alignment_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0 if temperature is not None else None,
                stop=stop,
                progress=progress,
                capture_usage=True,
                expected_json=True,
            )
            usage_totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
            usage_totals["completion_tokens"] += usage.get("completion_tokens", 0)

            try:
                alignment_obj = _json_loads_relaxed(alignment_content)
                aligned = bool(alignment_obj.get("aligned", False))
                assessment = (alignment_obj.get("assessment", "") or "").strip()
            except Exception:
                aligned = False
                assessment = "Failed to parse alignment response."

            # Boost score if aligned
            score += 1.0 if aligned else 0.0

        return score, aligned, assessment, errors, usage_totals

    def _expand_level(
        level: int,
        parent_best: Optional[ThoughtNode],
        total_usage: Dict[str, int],
    ) -> List[Tuple[str, str]]:
        """Ask the LLM for K candidate (model, data) pairs. Returns list of tuples."""
        parent_model = parent_best.model_code if parent_best else None
        parent_data = parent_best.data_code if parent_best else None
        expand_prompt = _build_tot_expand_prompt(
            prompt=prompt,
            grammar_implementation=grammar_implementation,
            few_shots=few_shots_list,
            k=TOT_BRANCH_FACTOR,
            parent_model=parent_model,
            parent_data=parent_data,
        )
        _notify(progress, f"[Level {level}] Expanding {TOT_BRANCH_FACTOR} thoughts")
        content, usage = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=expand_prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            stop=stop,
            progress=progress,
            capture_usage=True,
            expected_json=True,
        )
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

        if not content:
            raise RuntimeError("Empty expansion response.")
        try:
            arr = _json_loads_relaxed(content)

            # Robust fallbacks: sometimes models return an object wrapper
            if isinstance(arr, dict):
                if isinstance(arr.get("candidates"), list):
                    arr = arr["candidates"]
                elif "model" in arr and "data" in arr:
                    arr = [arr]

            if not isinstance(arr, list):
                raise ValueError("Expected a JSON array of candidate objects.")
            pairs: List[Tuple[str, str]] = []
            for obj in arr:
                if not isinstance(obj, dict):
                    continue
                m = obj.get("model")
                d = obj.get("data")
                if isinstance(m, str) and isinstance(d, str):
                    pairs.append((m, d))
            if not pairs:
                raise ValueError("No valid candidates found in expansion output.")
            # If fewer than requested, proceed with what we have
            return pairs[:TOT_BRANCH_FACTOR]
        except Exception as e:
            raise RuntimeError(f"Failed to parse expansion response as JSON: {e}\nResponse: {content}")

    # Track total token usage
    total_prompt_tokens = 0
    total_completion_tokens = 0

    best_overall: Optional[ThoughtNode] = None
    beam: List[ThoughtNode] = []  # best nodes at current level
    iterations_used = 0

    # Ensure output folders exist before writes
    model_dir = os.path.dirname(model_file)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    data_dir = os.path.dirname(data_file)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    for level in range(1, depth_limit + 1):
        iterations_used = level
        # Expand from best parent (greedy guided) or from prompt initially
        parent_for_expand = beam[0] if beam else best_overall
        usage_acc = {"prompt_tokens": 0, "completion_tokens": 0}
        candidate_pairs = _expand_level(level, parent_for_expand, usage_acc)
        total_prompt_tokens += usage_acc["prompt_tokens"]
        total_completion_tokens += usage_acc["completion_tokens"]

        # Score each candidate
        scored: List[ThoughtNode] = []
        for m_code, d_code in candidate_pairs:
            s, aligned, assess, errs, u = _score_candidate(m_code, d_code)
            total_prompt_tokens += u.get("prompt_tokens", 0)
            total_completion_tokens += u.get("completion_tokens", 0)
            scored.append(
                ThoughtNode(
                    model_code=m_code,
                    data_code=d_code,
                    score=s,
                    aligned=aligned,
                    assessment=assess,
                    syntax_errors=errs,
                    level=level,
                )
            )

        # Merge with previous beam to allow carrying forward the best-so-far (optional)
        pool = scored + (beam if beam else [])
        pool.sort(key=lambda n: n.score, reverse=True)
        beam = pool[:TOT_BEAM_WIDTH]
        top = beam[0] if beam else None

        if top:
            # Persist current best to disk (like previous API behavior)
            with open(model_file, "w") as f:
                f.write(top.model_code)
            with open(data_file, "w") as f:
                f.write(top.data_code)
            _notify(progress, f"[Level {level}] Wrote best candidate: {model_file} • {data_file} (score={top.score:.2f})")

            # Track best overall
            if best_overall is None or top.score > best_overall.score:
                best_overall = top

            # Early stop if compiled and aligned (or alignment disabled but compiled)
            if len(top.syntax_errors) == 0 and (top.aligned or not do_alignment):
                _notify(progress, "Solution meets criteria ✓ Stopping.")
                break
        else:
            _notify(progress, f"[Level {level}] No valid candidates; continuing")

    # Read latest attempt from disk (best written at last level)
    with open(model_file, "r") as f:
        model_code = f.read()
    with open(data_file, "r") as f:
        data_code = f.read()

    # Prepare final assessment text and syntax errors
    assessment_text = ""
    syntax_errors: List[str] = []
    if best_overall:
        assessment_text = best_overall.assessment or ""
        syntax_errors = best_overall.syntax_errors or []

    # Final assessment if failed or if alignment disabled and we still want a qualitative review
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
            "iterations": iterations_used,
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
    progress: Optional[Callable[[str], None]] = None,
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

    with open(model_file, "r") as fh:
        model_code = fh.read()
    with open(data_file, "r") as fh:
        data_code = fh.read()

    _notify(progress, "Generating feedback from LLM")
    user_prompt = _build_feedback_prompt(prompt, grammar_implementation, model_code, data_code)

    content: str = _llm_generate_text(
        provider=provider,
        model_name=model_name,
        input_text=user_prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0 if temperature is not None else None,
        stop=stop,
        progress=progress,
        capture_usage=False,
        expected_json=True,
    )
    if not content:
        raise RuntimeError("Empty model response.")
    try:
        _notify(progress, "Feedback received; parsing")
        return _json_loads_relaxed(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse feedback response as JSON: {e}\nResponse: {content}")


# ---------- Prompt builders ----------


def _build_tot_expand_prompt(
    prompt: str,
    grammar_implementation: str,
    few_shots: Optional[List[Dict[str, str]]] = None,
    k: int = TOT_BRANCH_FACTOR,
    parent_model: Optional[str] = None,
    parent_data: Optional[str] = None,
) -> str:
    """
    Tree-of-Thoughts expansion prompt: ask the LLM to return K diverse candidate (model, data) pairs.
    If parent_model/data are provided, request refinements/improvements over the parent.
    """
    few_shots_section = ""
    if few_shots:
        blocks: List[str] = []
        for i, ex in enumerate(few_shots, 1):
            desc_hdr = f'<description path="{ex.get("desc_path", "")}">'
            mod_hdr = f'<model_file path="{ex.get("model_path", "")}">'
            dat_hdr = f'<data_file path="{ex.get("data_path", "")}">'
            blocks.append(
                f'<example index="{i}">\n'
                f"{desc_hdr}\n{ex.get('description', '')}\n</description>\n\n"
                f"{mod_hdr}\n{ex.get('model', '')}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex.get('data', '')}\n</data_file>\n"
                f"</example>\n"
            )
        few_shots_section = (
            "<few_shot_examples>\n"
            "Use these exemplars for structure and syntax only; adapt names/indices to the new task.\n"
            + "".join(blocks)
            + "</few_shot_examples>\n\n"
        )

    parent_section = ""
    if parent_model or parent_data:
        parent_section = (
            "<parent_candidate>\n"
            "<note>Propose diverse, improved alternatives; fix issues if any; keep semantics consistent.</note>\n"
            f"<model>\n{parent_model or ''}\n</model>\n\n"
            f"<data>\n{parent_data or ''}\n</data>\n"
            "</parent_candidate>\n\n"
        )

    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        f"Generate {k} diverse candidate PyOPL model (.mod) and matching data (.dat) pairs for the problem.\n"
        "Think in a private scratchpad, but output only the final JSON response.\n"
        "Each candidate must be complete and self-consistent; ensure indices/domains and data consistency are correct.\n"
        "If any data are missing, create a small plausible instance.\n"
        "Candidates should be diverse in modeling choices or structures.\n"
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
        f"{parent_section}"
        "<output_requirements>\n"
        f'- Return ONLY a JSON array of exactly {k} objects. Each object must have keys "model" and "data".\n'
        "- Values must be valid JSON strings containing the full file contents. Use standard JSON escaping only.\n"
        "- Do not include the scratchpad or any additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON array in a ```json fenced block that contains only the JSON array.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        '{ "type": "array", "minItems": 1, "items": {\n'
        '  "type": "object", "additionalProperties": false,\n'
        '  "required": ["model","data"],\n'
        '  "properties": { "model": {"type":"string"}, "data": {"type":"string"} }\n'
        "} }\n"
        "</json_schema>\n"
        "<hint>\n"
        "Think step by step in the scratchpad, but output only the JSON array of candidates.\n"
        "</hint>\n"
    )
