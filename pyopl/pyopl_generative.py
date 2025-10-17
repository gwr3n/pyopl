# === Standard library imports ===
import inspect
import json
import logging
import os
import re
from enum import Enum, auto
from pathlib import Path  # NEW
from time import sleep
from typing import (
    Any,
    Callable,  # NEW
    Dict,
    List,  # NEW
    Optional,
    Tuple,  # NEW
)

# === Local imports ===
from .pyopl_core import OPLCompiler, SemanticError
from .rag_helper import rank_problem_descriptions as rag_rank  # NEW

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
    with open(path, "r") as f:
        return f.read()


def _read_pyopl_GBNF() -> str:
    grammar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "grammars", "PyOPL_GBNF")
    return _read_file(grammar_path)


def _read_pyopl_grammar() -> str:
    grammar_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "grammars", "PyOPL grammar.md")
    return _read_file(grammar_path)


def _read_pyopl_code() -> str:
    code_path = os.path.join(os.path.dirname(__file__), "pyopl_core.py")
    return _read_file(code_path)


def _get_grammar_implementation(mode: Grammar) -> str:
    if mode == Grammar.NONE:
        return ""
    if mode == Grammar.BNF:
        return _read_pyopl_grammar()
    if mode == Grammar.CODE:
        return _read_pyopl_code()
    raise ValueError(f"Invalid mode: {mode}")


# NEW: RAG few-shot helpers
def _safe_read_text(path: Path, max_chars: int = FEW_SHOT_MAX_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_chars:
            text = text[:max_chars]
        return text.strip()
    except Exception:
        return ""


def _find_pair_in_folder(desc_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Given a description .txt path, locate associated .mod and .dat in the same folder.
    Preference order:
      1) Same stem: <stem>.mod and <stem>.dat
      2) First *.mod and first *.dat in folder (sorted)
    """
    folder = desc_path.parent
    stem = desc_path.stem

    mod: Optional[Path] = folder / f"{stem}.mod"
    dat: Optional[Path] = folder / f"{stem}.dat"

    if not (mod and mod.exists() and mod.is_file()):
        mods = sorted(folder.glob("*.mod"))
        mod = mods[0] if mods else None

    if not (dat and dat.exists() and dat.is_file()):
        dats = sorted(folder.glob("*.dat"))
        dat = dats[0] if dats else None

    return (mod if mod and mod.exists() else None, dat if dat and dat.exists() else None)


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
    examples: List[Dict[str, str]] = []
    try:
        _notify(progress, f"Retrieving few-shot examples (k={k})")
        hits = rag_rank(query=problem_description, models_dir=models_dir, top_k=k)
        _notify(progress, f"Found {len(hits)} few-shot candidates: {[Path(hit['path']).name for hit in hits]}")
    except Exception as e:
        logger.debug(f"Few-shot retrieval skipped: {e}")
        _notify(progress, "Few-shot retrieval failed; continuing without examples")
        return examples

    for hit in hits:
        try:
            desc_path = Path(hit["path"])
            desc_text = _safe_read_text(desc_path)
            mod_path, dat_path = _find_pair_in_folder(desc_path)
            if not desc_text or not mod_path or not dat_path:
                continue
            mod_text = _safe_read_text(mod_path)
            dat_text = _safe_read_text(dat_path)
            if not mod_text or not dat_text:
                continue
            examples.append(
                {
                    "description": desc_text,
                    "model": mod_text,
                    "data": dat_text,
                    "desc_path": str(desc_path),
                    "model_path": str(mod_path),
                    "data_path": str(dat_path),
                }
            )
            if len(examples) >= k:
                break
        except Exception as e:
            logger.debug(f"Skipping example due to error: {e}")
            continue
    return examples


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
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai is not installed. pip install openai") from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def _google_client():
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google.generativeai is not installed. pip install google-generativeai") from e
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    return genai


def _ollama_generate_text(model_name: str, prompt: str, num_predict: Optional[int] = MAX_OUTPUT_TOKENS) -> str:
    """
    Call Ollama's Python client and return the response text.
    """
    try:
        from ollama import generate as ollama_generate
    except Exception as e:
        raise RuntimeError("ollama package is not installed. pip install ollama") from e
    options: Dict[str, Any] = {}
    if num_predict is not None:
        options["num_predict"] = num_predict
    resp = ollama_generate(model=model_name, prompt=prompt, options=options)
    try:
        return resp["response"] or ""
    except (TypeError, KeyError) as e:
        raise RuntimeError(f"Failed to retrieve response text from Ollama response: {e}")


def _build_create_params(
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model_name,
        "input": input_text,
        "response_format": {"type": "json"},
    }
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if stop:
        params["stop"] = stop
    return params


def _infer_provider(llm_provider: Optional[str], model_name: str) -> LLMProvider:
    if llm_provider:
        lp = llm_provider.strip().lower()
        if lp in ("openai", "oai"):
            return LLMProvider.OPENAI
        if lp in ("google", "genai", "gemini", "google.generativeai"):
            return LLMProvider.GOOGLE
        if lp in ("ollama",):
            return LLMProvider.OLLAMA
    # Heuristics by model name
    if model_name.startswith("gemini"):
        return LLMProvider.GOOGLE
    if "gpt-oss" in model_name or model_name.startswith(("llama", "qwen", "mistral")):
        return LLMProvider.OLLAMA
    return LLMProvider.OPENAI


def _llm_generate_text(
    provider: LLMProvider,
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    temperature: Optional[float] = None,
    stop: Optional[list[str]] = None,
    progress: Optional[Callable[[str], None]] = None,  # NEW
) -> str:
    if provider == LLMProvider.OPENAI:
        client = _openai_client()
        create_params = _build_create_params(
            model_name=model_name,
            input_text=input_text,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        _notify(progress, f"[LLM] OpenAI • {model_name}: sending request")  # NEW
        response = _call_openai_with_retry(client, create_params, progress=progress)  # CHANGED
        _notify(progress, "[LLM] OpenAI: response received")  # NEW
        response_text = _coalesce_response_text(response)
        if not response_text:
            raise RuntimeError(f"Empty OpenAI response: {response}.")
        return response_text

    if provider == LLMProvider.GOOGLE:
        genai = _google_client()
        model = genai.GenerativeModel(model_name)
        generation_config: Dict[str, Any] = {}
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        _notify(progress, f"[LLM] Gemini • {model_name}: sending request")  # NEW
        resp = model.generate_content(input_text, generation_config=generation_config)
        _notify(progress, "[LLM] Gemini: response received")  # NEW
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = []
            for c in resp.candidates:
                content = getattr(c, "content", None)
                if content and hasattr(content, "parts"):
                    for p in content.parts:
                        if hasattr(p, "text"):
                            parts.append(p.text or "")
            text = "".join(parts)
        return text or ""

    if provider == LLMProvider.OLLAMA:
        _notify(progress, f"[LLM] Ollama • {model_name}: generating")  # NEW
        result = _ollama_generate_text(model_name=model_name, prompt=input_text, num_predict=max_tokens)
        _notify(progress, "[LLM] Ollama: response received")  # NEW
        return result

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _call_openai_with_retry(
    client,
    create_params: Dict[str, Any],
    retries: int = 3,
    backoff_sec: float = 1.5,
    progress: Optional[Callable[[str], None]] = None,  # NEW
) -> Any:
    """
    Call Responses API with simple exponential backoff.
    Falls back by stripping newer/unsupported kwargs not supported by older SDKs/servers/models.
    """
    last_err: Optional[Exception] = None
    fallback_keys = ["response_format", "stop", "reasoning", "temperature"]
    params = dict(create_params)  # work on a copy

    # Prune params that the bound create() does not accept to avoid an initial TypeError
    try:
        create_callable = client.responses.create
        sig = inspect.signature(create_callable)
        # If create() accepts **kwargs (VAR_KEYWORD) then leave params as-is
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            supported = set(sig.parameters.keys())
            supported.discard("self")
            for k in list(params.keys()):
                if k not in supported:
                    params.pop(k, None)
    except Exception:
        # If introspection fails, fall back to the existing runtime-stripping logic below
        pass

    def _strip_param_from_error_message(msg: str) -> bool:
        removed = False
        low = msg.lower()
        for key in list(fallback_keys):
            if key in params and (
                f"unexpected keyword argument '{key}'" in low or f"unsupported parameter: '{key}'" in low or key in low
            ):
                params.pop(key, None)
                removed = True
        m = re.search(r"unsupported parameter:\s*'([^']+)'", msg, re.IGNORECASE)
        if m and m.group(1) in params:
            params.pop(m.group(1), None)
            removed = True
        m2 = re.search(r"parameter\s*'([^']+)'\s*is not supported", msg, re.IGNORECASE)
        if m2 and m2.group(1) in params:
            params.pop(m2.group(1), None)
            removed = True
        return removed

    for attempt in range(retries):
        try:
            return client.responses.create(**params)
        except Exception as e:
            last_err = e
            msg = str(e) if e else "unknown error"
            _notify(progress, f"[LLM] OpenAI: {msg}")
            if _strip_param_from_error_message(msg):
                _notify(progress, "[LLM] OpenAI: retrying without unsupported parameters")  # NEW
                continue
            _notify(progress, f"[LLM] OpenAI: retry {attempt + 1}/{retries} after error: {msg}")  # NEW
            sleep(backoff_sec * (2**attempt))
    _notify(progress, f"[LLM] OpenAI: failed after {retries} attempts")  # NEW
    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_err}")


# ---------- Prompt builders ----------


def _build_generation_prompt(
    prompt: str, grammar_implementation: str, few_shots: Optional[List[Dict[str, str]]] = None
) -> str:  # CHANGED
    few_shots_section = ""
    if few_shots:
        blocks = []
        for i, ex in enumerate(few_shots, 1):
            # Include file paths as metadata to provide provenance (optional for the model)
            desc_hdr = f'<description path="{ex.get("desc_path", "")}">'
            mod_hdr = f'<model_file path="{ex.get("model_path", "")}">'
            dat_hdr = f'<data_file path="{ex.get("data_path", "")}">'
            blocks.append(
                f'<example index="{i}">\n'
                f"{desc_hdr}\n{ex['description']}\n</description>\n\n"
                f"{mod_hdr}\n{ex['model']}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex['data']}\n</data_file>\n"
                f"</example>\n"
            )
        few_shots_section = (
            "<few_shot_examples>\n"
            "Use the following exemplars as guidance for structure and syntax only. Do not copy variable names unless appropriate to the new problem.\n"
            + "".join(blocks)
            + "</few_shot_examples>\n\n"
        )

    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Generate a valid PyOPL model (.mod) and a matching data file (.dat) for the given problem description.\n"
        "Ensure the model decision variables, objective function, and constraints fully align with the provided problem description.\n"
        "Infer from the problem context whether decision variables should be integer, binary, or continuous.\n"
        "Label all constraints and the objective function meaningfully; "
        "thoroughly comment the model to explain the purpose of variables, parameters, objective, and constraints; "
        "match these explanations to the problem description by following the predicaments of literate programming.\n"
        "If data are missing, create a small, plausible mock instance consistent with the model.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "You are also provided with a few-shot examples section; use it only as guidance and produce a solution tailored to the new problem.\n"
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
        '- Return ONLY a JSON object with exactly two keys: "model" (the PyOPL model) and "data" (the matching data file).\n'
        "- The values must be single JSON strings (no arrays/objects inside them).\n"
        "- Escape all double quotes and backslashes; encode newlines as \\n.\n"
        "- No trailing commas. No additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["model", "data"],\n'
        '  "properties": {\n'
        '    "model": {"type": "string"},\n'
        '    "data":  {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        "{\n"
        '  "model": "float a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
        '  "data":  "a = 10;\\nb= 5;"\n'
        "}\n"
        "</example_output>\n"
    )


def _build_alignment_prompt(prompt: str, grammar_implementation: str, model_code: str, data_code: str) -> str:
    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Assess whether the generated PyOPL model and data fully align with the problem description.\n"
        "Alignment means the objective, constraints, decision variables, and data match the problem description.\n"
        "Be critical and specific about modeling choices, feasibility, and consistency.\n"
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
        "- No Markdown. No bullet lists. No commentary. No additional keys. No trailing commas.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
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


def _build_revision_prompt_alignment(
    prompt: str,
    grammar_implementation: str,
    assessment_text: str,
    model_code: str,
    data_code: str,  # CHANGED
    few_shots: Optional[List[Dict[str, str]]] = None,  # NEW
) -> str:
    few_shots_section = ""
    if few_shots:
        blocks = []
        for i, ex in enumerate(few_shots, 1):
            desc_hdr = f'<description path="{ex.get("desc_path", "")}">'
            mod_hdr = f'<model_file path="{ex.get("model_path", "")}">'
            dat_hdr = f'<data_file path="{ex.get("data_path", "")}">'
            blocks.append(
                f'<example index="{i}">\n'
                f"{desc_hdr}\n{ex['description']}\n</description>\n\n"
                f"{mod_hdr}\n{ex['model']}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex['data']}\n</data_file>\n"
                f"</example>\n"
            )
        few_shots_section = (
            "<few_shot_examples>\n"
            "Use the following exemplars as guidance for structure and syntax only. Do not copy variable names unless appropriate to the new problem.\n"
            + "".join(blocks)
            + "</few_shot_examples>\n\n"
        )

    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "The previous attempt produced a syntactically valid PyOPL model and data, but they are NOT fully aligned with the problem description.\n"
        "<assessment>\n"
        f"{assessment_text}\n"
        "</assessment>\n"
        "Revise the model and data so that they fully align with the problem description while preserving syntactic validity.\n"
        "Change only what is necessary to achieve alignment (objective, constraints, variables, sets/parameters, and data consistency).\n"
        "Label all constraints and the objective function meaningfully; "
        "thoroughly comment the model to explain the purpose of variables, parameters, objective, and constraints; "
        "match these explanations to the problem description by following the predicaments of literate programming.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "You are also provided with a few-shot examples section; use it only as guidance and produce a solution tailored to the new problem.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        f"{few_shots_section}"  # NEW
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<previous_attempt>\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n"
        "</previous_attempt>\n\n"
        "<revision_guidelines>\n"
        "- Ensure the objective, constraints, indices, and variable domains reflect the problem description.\n"
        "- Make the minimal set of changes necessary to correct misalignment.\n"
        "- Keep syntax strictly valid.\n"
        "- Return complete model and data strings; do not return diffs.\n"
        "</revision_guidelines>\n\n"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with exactly two keys: "model" (the PyOPL model) and "data" (the matching data file).\n'
        "- The values must be single JSON strings (no arrays/objects inside them).\n"
        "- Escape all double quotes and backslashes; encode newlines as \\n.\n"
        "- No trailing commas. No additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["model", "data"],\n'
        '  "properties": {\n'
        '    "model": {"type": "string"},\n'
        '    "data":  {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        "{\n"
        '  "model": "float a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
        '  "data":  "a = 10;\\nb= 5;"\n'
        "}\n"
        "</example_output>\n"
    )


def _build_revision_prompt_syntax(
    prompt: str,
    grammar_implementation: str,
    model_code: str,
    data_code: str,
    syntax_errors,  # CHANGED
    few_shots: Optional[List[Dict[str, str]]] = None,  # NEW
) -> str:
    few_shots_section = ""
    if few_shots:
        blocks = []
        for i, ex in enumerate(few_shots, 1):
            desc_hdr = f'<description path="{ex.get("desc_path", "")}">'
            mod_hdr = f'<model_file path="{ex.get("model_path", "")}">'
            dat_hdr = f'<data_file path="{ex.get("data_path", "")}">'
            blocks.append(
                f'<example index="{i}">\n'
                f"{desc_hdr}\n{ex['description']}\n</description>\n\n"
                f"{mod_hdr}\n{ex['model']}\n</model_file>\n\n"
                f"{dat_hdr}\n{ex['data']}\n</data_file>\n"
                f"</example>\n"
            )
        few_shots_section = (
            "<few_shot_examples>\n"
            "Use the following exemplars as guidance for structure and syntax only. Do not copy variable names unless appropriate to the new problem.\n"
            + "".join(blocks)
            + "</few_shot_examples>\n\n"
        )

    return (
        "<role>\n"
        "You are an expert in mathematical optimization and PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "The previous attempt to generate a PyOPL model and data file failed due to syntax errors.\n"
        "Revise the model and data to fix the errors while retaining alignment with the problem description.\n"
        "Change only what is necessary to fix the errors.\n"
        "Label all constraints and the objective function meaningfully; "
        "thoroughly comment the model to explain the purpose of variables, parameters, objective, and constraints; "
        "match these explanations to the problem description by following the predicaments of literate programming.\n"
        "Use the following PyOPL syntax implementation as a reference for valid PyOPL syntax.\n"
        "You are also provided with a few-shot examples section; use it only as guidance and produce a solution tailored to the new problem.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar_implementation}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        f"{few_shots_section}"  # NEW
        "<problem_description>\n"
        f"{prompt}\n"
        "</problem_description>\n\n"
        "<previous_attempt>\n"
        "<model>\n"
        f"{model_code}\n"
        "</model>\n\n"
        "<data>\n"
        f"{data_code}\n"
        "</data>\n"
        "</previous_attempt>\n\n"
        "<errors>\n"
        f"{syntax_errors}\n"
        "</errors>\n\n"
        "<revision_guidelines>\n"
        "- Fix the listed syntax/semantic errors.\n"
        "- Preserve the original modeling intent and structure when possible.\n"
        "- Ensure the model compiles with the data under the given implementation.\n"
        "- Return complete model and data strings; do not return diffs.\n"
        "</revision_guidelines>\n\n"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with exactly two keys: "model" (the PyOPL model) and "data" (the matching data file).\n'
        "- The values must be single JSON strings (no arrays/objects inside them).\n"
        "- Escape all double quotes and backslashes; encode newlines as \\n.\n"
        "- No trailing commas. No additional keys. No commentary.\n"
        "- Optional: you MAY wrap the JSON in a ```json fenced block; if you do, the fence must contain only the JSON.\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        "{\n"
        '  "type": "object",\n'
        '  "additionalProperties": false,\n'
        '  "required": ["model", "data"],\n'
        '  "properties": {\n'
        '    "model": {"type": "string"},\n'
        '    "data":  {"type": "string"}\n'
        "  }\n"
        "}\n"
        "</json_schema>\n\n"
        "<example_output>\n"
        "{\n"
        '  "model": "float a;\\nfloat b;\\ndvar float x;\\nminimize z: a*x;\\nsubject to { b*x >= 0; }",'
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
        "Assess how well the generated PyOPL model and data align with the problem description.\n"
        "Be critical and specific about modeling choices, feasibility, and consistency.\n"
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
    few_shot: bool = True,
):
    """Generate a PyOPL model and data file from a prompt using OpenAI GPT, validate with pyopl, iterate on errors, and assess alignment.

    Args:
        prompt (str): Problem description to model.
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
        progress (callable|None): Optional function that receives progress messages (str).  # NEW

    Returns:
        str or dict: If return_statistics is False, returns the final assessment string.
                     If return_statistics is True, returns a dict with keys:
                     - "iterations": number of iterations performed
                     - "assessment": final assessment string
                     - "syntax_errors": list of syntax errors encountered (if any)
    Raises:
        RuntimeError: If generation or validation fails irrecoverably.
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
        f"Generating with provider={provider.value} model={model_name} iterations={iterations} alignment={'on' if do_alignment else 'off'}",
    )  # NEW

    # NEW: Retrieve few-shot examples using RAG
    few_shots: List[Dict[str, str]] = (
        _gather_few_shots(prompt, k=FEW_SHOT_TOP_K, models_dir="opl_models", progress=progress) if few_shot else []
    )

    user_prompt = _build_generation_prompt(prompt, grammar_implementation, few_shots=few_shots)  # CHANGED
    assessment_text = ""
    syntax_errors: list[str] = []
    model_code = ""
    data_code = ""

    for iteration in range(iterations):
        logger.debug(f"Iteration {iteration + 1}/{iterations}")
        _notify(progress, f"Iteration {iteration + 1}/{iterations}: prompting model")  # NEW
        content = _llm_generate_text(
            provider=provider,
            model_name=model_name,
            input_text=user_prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=temperature,
            stop=stop,
            progress=progress,  # NEW
        )
        if not content:
            raise RuntimeError("Empty model response.")
        try:
            result = _json_loads_relaxed(content)
            model_code = result["model"]
            data_code = result["data"]
            _notify(progress, "LLM response parsed (model + data)")  # NEW
            logger.debug("Model and data generated.")
        except Exception as e:
            raise RuntimeError(f"Failed to parse model response as JSON: {e}\nResponse: {content}")

        compiler = OPLCompiler()
        syntax_errors = []
        try:
            _notify(progress, "Compiling model and data")  # NEW
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
        _notify(progress, f"Wrote files: {model_file} • {data_file}")  # NEW

        if not syntax_errors:
            if not do_alignment:
                _notify(progress, "Syntax OK; alignment check disabled. Stopping.")  # NEW
                break

            logger.debug("Checking alignment with original prompt...")
            _notify(progress, "Checking alignment with original prompt...")  # NEW
            alignment_prompt = _build_alignment_prompt(prompt, grammar_implementation, model_code, data_code)
            alignment_content = _llm_generate_text(
                provider=provider,
                model_name=model_name,
                input_text=alignment_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0 if temperature is not None else None,
                stop=stop,
                progress=progress,  # NEW
            )
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
                    _notify(progress, "Aligned ✓ Stopping.")  # NEW
                    logger.debug("Model and data are syntactically valid and aligned with the prompt.")
                    break
                else:
                    _notify(progress, "Not aligned; revising per assessment")  # NEW
                    logger.debug(
                        f"Model and data are syntactically valid but NOT aligned with the prompt. Assessment: {assessment_text}"
                    )
                    user_prompt = _build_revision_prompt_alignment(  # CHANGED
                        prompt, grammar_implementation, assessment_text, model_code, data_code, few_shots=few_shots  # NEW
                    )
            else:
                raise RuntimeError(f"Invalid alignment response JSON: {alignment_content}")
        else:
            _notify(progress, f"Syntax/semantic errors found: {len(syntax_errors)}; revising...")  # NEW
            logger.debug("Model or data has syntax errors; revising...")
            user_prompt = _build_revision_prompt_syntax(  # CHANGED
                prompt, grammar_implementation, model_code, data_code, syntax_errors, few_shots=few_shots  # NEW
            )

    # Load latest version of the model and data files
    with open(model_file, "r") as f:
        model_code = f.read()
    with open(data_file, "r") as f:
        data_code = f.read()

    if syntax_errors or not do_alignment:
        logger.debug("Final assessment of model and data alignment...")
        _notify(progress, "Requesting final assessment")  # NEW
        assessment_prompt = _build_final_assessment_prompt(
            prompt, grammar_implementation, model_code, data_code, syntax_errors
        )
        assessment_text = (
            _llm_generate_text(
                provider=provider,
                model_name=model_name,
                input_text=assessment_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2 if temperature is not None else None,
                stop=stop,
                progress=progress,  # NEW
            )
            or ""
        )

    _notify(progress, "Generation complete")  # NEW
    if return_statistics:
        return {
            "iterations": iteration + 1,
            "assessment": assessment_text.strip(),
            "syntax_errors": syntax_errors,
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

    content = _llm_generate_text(
        provider=provider,
        model_name=model_name,
        input_text=user_prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0 if temperature is not None else None,
        stop=stop,
        progress=progress,  # NEW
    )
    if not content:
        raise RuntimeError("Empty model response.")
    try:
        _notify(progress, "Feedback received; parsing")  # NEW
        return _json_loads_relaxed(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse feedback response as JSON: {e}\nResponse: {content}")


# ---------- Model discovery ----------


def list_openai_models(prefix: Optional[str] = "gpt") -> list[str]:
    """
    Return available OpenAI model IDs visible to the API key.
    Optionally filter by prefix.
    """
    client = _openai_client()
    try:
        resp = client.models.list()
    except Exception as e:
        raise RuntimeError(f"Failed to list OpenAI models: {e}")

    names: list[str] = []
    # Support both SDK return shapes
    data = getattr(resp, "data", None)
    items = data if isinstance(data, list) else (list(resp) if resp is not None else [])
    for m in items:
        mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
        if isinstance(mid, str):
            names.append(mid)
    if prefix:
        names = [n for n in names if n.startswith(prefix)]
    return sorted(set(names))


def list_gemini_models(prefix: Optional[str] = "gemini") -> list[str]:
    """
    Return available Google Generative AI model names.
    By default, returns models starting with 'gemini' and supporting generateContent.
    """
    genai = _google_client()
    try:
        models = genai.list_models()
    except Exception as e:
        raise RuntimeError(f"Failed to list Gemini models: {e}")

    names: list[str] = []
    for m in models or []:
        if "generateContent" in m.supported_generation_methods:
            name = m.name
            if isinstance(name, str) and name.startswith("models/"):
                name = name[len("models/") :]
                if prefix and name.startswith(prefix):
                    names.append(name)

    return sorted(set(names))


def list_ollama_models(prefix: Optional[str] = None) -> list[str]:
    """
    Return available local Ollama model tags (e.g., 'llama3:8b-instruct').
    """
    try:
        from ollama import list as ollama_list
    except Exception as e:
        raise RuntimeError("ollama package is not installed. pip install ollama") from e

    models: list[str] = []
    try:
        resp = ollama_list()

        for model in resp["models"]:
            models.append(model.model)

    except Exception as e:
        raise RuntimeError(f"Failed to list Ollama models: {e}")

    if prefix:
        models = [n for n in models if n.startswith(prefix)]
    return sorted(set(models))


def list_models(llm_provider: Optional[str] = None, model_name: str = MODEL_NAME) -> list[str]:
    """
    Unified helper: returns models for the inferred provider.
    llm_provider: 'openai', 'google', or 'ollama' (None -> inferred from model_name).
    """
    provider = _infer_provider(llm_provider, model_name)
    if provider == LLMProvider.OPENAI:
        return list_openai_models()
    if provider == LLMProvider.GOOGLE:
        return list_gemini_models()
    if provider == LLMProvider.OLLAMA:
        return list_ollama_models()
    raise ValueError(f"Unsupported LLM provider: {provider}")
