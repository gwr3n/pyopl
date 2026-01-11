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
from ._strategy_base import (
    list_gemini_models as _base_list_gemini_models,
)
from ._strategy_base import (
    list_models as _base_list_models,
)
from ._strategy_base import (
    list_ollama_models as _base_list_ollama_models,
)
from ._strategy_base import (
    list_openai_models as _base_list_openai_models,
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

# NEW: CoE configuration
COE_FORWARD_STEPS = 5
COE_DEFAULT_EXPERTS = [
    "Terminology Interpreter",
    "Modeling Expert",
    "Data Builder",
    "Code Reviewer",
]


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


# ---------- Prompt builders (CoE) ----------


def _format_few_shots_knowledge(few_shots: List[Dict[str, str]]) -> str:
    if not few_shots:
        return ""
    blocks: List[str] = []
    for i, ex in enumerate(few_shots, 1):
        blocks.append(
            f'<example index="{i}">\n'
            f"<description path=\"{ex.get('desc_path','')}\">\n{ex.get('description','')}\n</description>\n"
            f"<model_file path=\"{ex.get('model_path','')}\">\n{ex.get('model','')}\n</model_file>\n"
            f"<data_file path=\"{ex.get('data_path','')}\">\n{ex.get('data','')}\n</data_file>\n"
            f"</example>\n"
        )
    return "<knowledge_base>\n" + "".join(blocks) + "</knowledge_base>\n"


def _build_conductor_prompt(problem: str, experts: List[str], comments: List[Dict[str, str]], remaining_steps: int) -> str:
    """
    Ask the conductor to pick the next expert from the provided list.
    Return JSON: { "next_expert": "<one of experts>", "reason": "<short>" }
    """
    comments_text = "\n".join([f"- {c['expert']}: {c['comment']}" for c in comments[-10:]])
    expert_list = ", ".join(experts)
    return (
        "<role>\n"
        "You are the Conductor coordinating experts to solve an optimization modeling task in PyOPL.\n"
        "</role>\n\n"
        "<task>\n"
        "Choose the next expert to consult to make progress toward a correct PyOPL model (.mod) and data (.dat).\n"
        f"You must pick ONE from: [{expert_list}].\n"
        "Be pragmatic. If modeling is unclear, consult Modeling. If data is missing, consult Data Builder.\n"
        "If terminology is unclear, consult Terminology Interpreter. If close to done, ask Code Reviewer.\n"
        f"Remaining steps: {remaining_steps}.\n"
        "</task>\n\n"
        "<problem>\n"
        f"{problem}\n"
        "</problem>\n\n"
        "<context>\n"
        f"{comments_text or '(no comments yet)'}\n"
        "</context>\n\n"
        "<output>\n"
        'Return ONLY JSON: {"next_expert": "<name from list>", "reason": "<short>"}\n'
        "</output>\n"
    )


def _build_expert_prompt(
    expert: str, problem: str, grammar: str, comments: List[Dict[str, str]], few_shots: List[Dict[str, str]]
) -> str:
    """
    Build prompts for individual experts. Output must be JSON: {"comment":"<short actionable insight or snippet>"}.
    Experts: Terminology Interpreter, Modeling Expert, Data Builder, Code Reviewer.
    """
    comments_text = "\n".join([f"- {c['expert']}: {c['comment']}" for c in comments[-10:]])
    kb = _format_few_shots_knowledge(few_shots)
    common_prefix = (
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        f"{kb}"
        "<problem>\n"
        f"{problem}\n"
        "</problem>\n\n"
        "<prior_comments>\n"
        f"{comments_text or '(none)'}\n"
        "</prior_comments>\n"
        "<output>\n"
        'Return ONLY JSON: {"comment":"<short actionable insight or snippet>"}\n'
        "</output>\n"
    )
    if expert == "Terminology Interpreter":
        return (
            "<role>Terminology Interpreter</role>\n"
            "<task>\n"
            "Clarify domain terms, implicit constraints, and edge conditions that affect modeling and data.\n"
            "Point out zero lead times, backlogging, capacity semantics, initial/final statuses, and objective components.\n"
            "Be concise and actionable.\n"
            "</task>\n\n" + common_prefix
        )
    if expert == "Modeling Expert":
        return (
            "<role>Modeling Expert (PyOPL)</role>\n"
            "<task>\n"
            "Propose a correct PyOPL MODEL structure: sets, parameters, decision variables (domains), objective, and constraints.\n"
            "If uncertain, state assumptions explicitly. Keep names consistent and ready for a Reducer to synthesize final .mod.\n"
            "Return a compact, well-structured outline or small code snippet (not the full model yet).\n"
            "</task>\n\n" + common_prefix
        )
    if expert == "Data Builder":
        return (
            "<role>Data Builder (PyOPL)</role>\n"
            "<task>\n"
            "Propose a consistent PyOPL DATA outline matching the current modeling assumptions.\n"
            "If data are missing, create a minimal plausible instance. Keep it small and consistent with indices.\n"
            "</task>\n\n" + common_prefix
        )
    if expert == "Code Reviewer":
        return (
            "<role>Code Reviewer (PyOPL)</role>\n"
            "<task>\n"
            "Identify likely issues or missing links between model and data, suggest minimal fixes or clarifications.\n"
            "Focus on variable domains, indices, constraint signs, bounds, and objective completeness.\n"
            "</task>\n\n" + common_prefix
        )
    # Fallback to modeling
    return _build_expert_prompt("Modeling Expert", problem, grammar, comments, few_shots)


def _build_reducer_prompt(problem: str, grammar: str, comments: List[Dict[str, str]], few_shots: List[Dict[str, str]]) -> str:
    """
    Reducer synthesizes final PyOPL model+data.
    Output must be JSON: {"model":"<.mod>", "data":"<.dat>"} (strings).
    """
    kb = _format_few_shots_knowledge(few_shots)
    comments_text = "\n".join([f"- {c['expert']}: {c['comment']}" for c in comments])
    return (
        "<role>Reducer (PyOPL)</role>\n"
        "<task>\n"
        "Synthesize a correct PyOPL model (.mod) and matching data (.dat) from the expert comments.\n"
        "Ensure consistency of sets, parameters, indices, variable domains, objectives, and constraints.\n"
        "The output MUST compile under the provided PyOPL grammar implementation.\n"
        "If assumptions were stated, reflect them coherently.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        f"{kb}"
        "<problem>\n"
        f"{problem}\n"
        "</problem>\n\n"
        "<comments>\n"
        f"{comments_text}\n"
        "</comments>\n\n"
        "<output_requirements>\n"
        '- Return ONLY a JSON object with exactly two keys: "model" and "data".\n'
        "- Each value must be a single JSON string (escape quotes/backslashes, encode newlines as \\n).\n"
        "</output_requirements>\n\n"
        "<json_schema>\n"
        '{ "type": "object", "additionalProperties": false,\n'
        '  "required": ["model", "data"],\n'
        '  "properties": { "model": {"type": "string"}, "data": {"type": "string"} } }\n'
        "</json_schema>\n"
    )


def _build_reflection_prompt(expert: str, problem: str, grammar: str, comments: List[Dict[str, str]], feedback: str) -> str:
    """
    Ask a specific expert to reflect and update their comment based on evaluator/conformance feedback.
    Output must be JSON: {"comment":"<updated>"}.
    """
    comments_text = "\n".join([f"- {c['expert']}: {c['comment']}" for c in comments[-10:]])
    return (
        f"<role>{expert} — Reflection</role>\n"
        "<task>\n"
        "Given evaluator feedback, correct or refine your previous advice.\n"
        "Be minimal but precise; resolve the pointed issue directly.\n"
        "</task>\n\n"
        "<grammar_reference>\n"
        "--- BEGIN PYOPL SYNTAX IMPLEMENTATION ---\n"
        f"{grammar}\n"
        "--- END PYOPL SYNTAX IMPLEMENTATION ---\n"
        "</grammar_reference>\n\n"
        "<problem>\n"
        f"{problem}\n"
        "</problem>\n\n"
        "<prior_context>\n"
        f"{comments_text or '(none)'}\n"
        "</prior_context>\n"
        "<evaluator_feedback>\n"
        f"{feedback}\n"
        "</evaluator_feedback>\n\n"
        "<output>\n"
        'Return ONLY JSON: {"comment":"<updated>"}\n'
        "</output>\n"
    )


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


# ---------- CoE Orchestration ----------


def _call_json(provider, model_name, prompt, progress, temperature=None, stop=None) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Call LLM expecting JSON; parse with relaxed loader. Returns (obj, usage).
    """
    content, usage = _llm_generate_text(
        provider=provider,
        model_name=model_name,
        input_text=prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        stop=stop,
        progress=progress,
        capture_usage=True,
    )
    if not content:
        raise RuntimeError("Empty LLM response.")
    try:
        return _json_loads_relaxed(content), usage
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM JSON response: {e}\nResponse: {content}")


def _run_chain_of_experts(
    problem: str,
    grammar_implementation: str,
    provider: LLMProvider,
    model_name: str,
    progress: Optional[Callable[[str], None]],
    temperature: Optional[float],
    stop: Optional[list[str]],
    few_shots: List[Dict[str, str]],
    max_forward_steps: int,
    max_trials: int,
    do_alignment: bool,
) -> Tuple[str, str, str, List[str], Dict[str, int]]:
    """
    Implements Conductor -> Experts -> Reducer -> Evaluator with backward reflection (Xiao et al., 2024).
    Returns (model_code, data_code, assessment_text, syntax_errors, usage_totals).
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    assessment_text = ""
    syntax_errors: List[str] = []
    model_code = ""
    data_code = ""

    experts_catalog = list(COE_DEFAULT_EXPERTS)

    trials_run = 0
    for trial in range(1, max_trials + 1):
        trials_run = trial
        _notify(progress, f"[CoE] Trial {trial}/{max_trials}: forward-thought construction")
        comments: List[Dict[str, str]] = []
        expert_stack: List[str] = []

        # Forward thought construction
        for step in range(1, max_forward_steps + 1):
            remaining = max_forward_steps - step + 1
            # Conductor picks expert
            try:
                conductor_prompt = _build_conductor_prompt(problem, experts_catalog, comments, remaining)
                conductor_obj, u = _call_json(
                    provider,
                    model_name,
                    conductor_prompt,
                    progress,
                    temperature=0.0 if temperature is not None else None,
                    stop=stop,
                )
                total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += u.get("completion_tokens", 0)
                next_expert = str(conductor_obj.get("next_expert") or "").strip()
                if next_expert not in experts_catalog:
                    # Fallback heuristic
                    next_expert = experts_catalog[min(step - 1, len(experts_catalog) - 1)]
            except Exception as e:
                _notify(progress, f"[CoE] Conductor fallback due to error: {e}")
                next_expert = experts_catalog[min(step - 1, len(experts_catalog) - 1)]

            # Query expert
            expert_prompt = _build_expert_prompt(next_expert, problem, grammar_implementation, comments, few_shots)
            try:
                expert_obj, u2 = _call_json(provider, model_name, expert_prompt, progress, temperature=temperature, stop=stop)
                total_usage["prompt_tokens"] += u2.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += u2.get("completion_tokens", 0)
                comment = str(expert_obj.get("comment") or "").strip()
                if comment:
                    comments.append({"expert": next_expert, "comment": comment})
                    expert_stack.append(next_expert)
                    _notify(progress, f"[CoE] {next_expert}: contributed")
            except Exception as e:
                _notify(progress, f"[CoE] {next_expert} failed: {e}")

        # Reducer synthesizes final model+data
        _notify(progress, "[CoE] Reducing comments into PyOPL model+data")
        reducer_prompt = _build_reducer_prompt(problem, grammar_implementation, comments, few_shots)
        try:
            reducer_obj, u3 = _call_json(provider, model_name, reducer_prompt, progress, temperature=temperature, stop=stop)
            total_usage["prompt_tokens"] += u3.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u3.get("completion_tokens", 0)
            model_code = str(reducer_obj["model"])
            data_code = str(reducer_obj["data"])
        except Exception as e:
            raise RuntimeError(f"Reducer failed to produce valid JSON: {e}")

        # Evaluate by compiling
        compiler = OPLCompiler()
        syntax_errors = []
        try:
            _notify(progress, "[CoE] Compiling model and data")
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            syntax_errors.append(str(e))
        except Exception as e:
            syntax_errors.append(f"{type(e).__name__}: {e}")

        # Write current attempt to disk
        # model_dir = os.path.dirname("noop")  # ensure file ops below safe even if no dir
        try:
            # The caller will write to desired files; we only return strings here
            pass
        except Exception:
            pass

        if not syntax_errors:
            # Optional alignment check
            if do_alignment:
                _notify(progress, "[CoE] Checking alignment with original prompt...")
                align_prompt = _build_alignment_prompt(problem, grammar_implementation, model_code, data_code)
                try:
                    align_obj, u4 = _call_json(
                        provider,
                        model_name,
                        align_prompt,
                        progress,
                        temperature=0.0 if temperature is not None else None,
                        stop=stop,
                    )
                    total_usage["prompt_tokens"] += u4.get("prompt_tokens", 0)
                    total_usage["completion_tokens"] += u4.get("completion_tokens", 0)
                    assessment_text = str(align_obj.get("assessment", "")).strip()
                    if bool(align_obj.get("aligned", False)):
                        _notify(progress, "[CoE] Compiled and aligned ✓")
                        break
                    else:
                        _notify(progress, "[CoE] Not aligned; starting reflection")
                        # Use alignment assessment as feedback
                        feedback_text = f"Alignment issues: {assessment_text}"
                except Exception as e:
                    _notify(progress, f"[CoE] Alignment check failed: {e}")
                    feedback_text = "Model may be misaligned with prompt."
            else:
                _notify(progress, "[CoE] Compiled ✓ (alignment disabled)")
                break
        else:
            _notify(progress, f"[CoE] Compilation failed with {len(syntax_errors)} error(s); starting reflection")
            feedback_text = "Syntax/semantic errors: " + "; ".join(syntax_errors)

        # Backward reflection
        _notify(progress, "[CoE] Backward reflection phase")
        stop_backward = False
        # Reflect in reverse order of expert contributions
        for ex in reversed(expert_stack):
            if stop_backward:
                break
            try:
                reflect_prompt = _build_reflection_prompt(ex, problem, grammar_implementation, comments, feedback_text)
                reflect_obj, u5 = _call_json(
                    provider,
                    model_name,
                    reflect_prompt,
                    progress,
                    temperature=0.0 if temperature is not None else None,
                    stop=stop,
                )
                total_usage["prompt_tokens"] += u5.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += u5.get("completion_tokens", 0)
                new_comment = str(reflect_obj.get("comment") or "").strip()
                if new_comment:
                    comments.append({"expert": ex, "comment": new_comment})
            except Exception as e:
                _notify(progress, f"[CoE] Reflection by {ex} failed: {e}")

        # After reflection, re-reduce once
        _notify(progress, "[CoE] Re-reducing after reflection")
        try:
            reducer_obj2, u6 = _call_json(
                provider,
                model_name,
                _build_reducer_prompt(problem, grammar_implementation, comments, few_shots),
                progress,
                temperature=temperature,
                stop=stop,
            )
            total_usage["prompt_tokens"] += u6.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u6.get("completion_tokens", 0)
            model_code = str(reducer_obj2["model"])
            data_code = str(reducer_obj2["data"])
        except Exception as e:
            _notify(progress, f"[CoE] Reducer after reflection failed: {e}")
            continue

        # Re-evaluate
        compiler = OPLCompiler()
        syntax_errors = []
        try:
            _notify(progress, "[CoE] Re-compiling model and data")
            compiler.compile_model(model_code, data_code)
        except SemanticError as e:
            syntax_errors.append(str(e))
        except Exception as e:
            syntax_errors.append(f"{type(e).__name__}: {e}")

        if not syntax_errors and do_alignment:
            try:
                align_obj2, u7 = _call_json(
                    provider,
                    model_name,
                    _build_alignment_prompt(problem, grammar_implementation, model_code, data_code),
                    progress,
                    temperature=0.0 if temperature is not None else None,
                    stop=stop,
                )
                total_usage["prompt_tokens"] += u7.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += u7.get("completion_tokens", 0)
                assessment_text = str(align_obj2.get("assessment", "")).strip()
                if bool(align_obj2.get("aligned", False)):
                    _notify(progress, "[CoE] Compiled and aligned ✓ after reflection")
                    break
                else:
                    _notify(progress, "[CoE] Still not aligned; continuing trials if any")
            except Exception as e:
                _notify(progress, f"[CoE] Alignment post-reflection failed: {e}")

        if not syntax_errors and not do_alignment:
            _notify(progress, "[CoE] Compiled ✓ after reflection (alignment disabled)")
            break

    return model_code, data_code, assessment_text, syntax_errors, total_usage, trials_run


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
    """Generate a PyOPL model and data file using Chain-of-Experts (CoE):
    forward thought construction (Conductor + Experts) -> Reducer -> compile (+ optional alignment),
    with backward reflection. Repeated up to `iterations` trials.

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
        f"CoE: provider={provider.value} model={model_name} trials={iterations} forward_steps={COE_FORWARD_STEPS} alignment={'on' if do_alignment else 'off'}",
    )

    # Few-shot examples (static per run)
    few_shots_list: List[Dict[str, str]] = (
        _gather_few_shots(prompt, k=FEW_SHOT_TOP_K, models_dir=None, progress=progress) if few_shot else []
    )

    # Run Chain-of-Experts
    model_code, data_code, assessment_text, syntax_errors, usage_totals, trials_run = _run_chain_of_experts(
        problem=prompt,
        grammar_implementation=grammar_implementation,
        provider=provider,
        model_name=model_name,
        progress=progress,
        temperature=temperature,
        stop=stop,
        few_shots=few_shots_list,
        max_forward_steps=COE_FORWARD_STEPS,
        max_trials=iterations,
        do_alignment=do_alignment,
    )

    # Ensure output folder exists and write final artifacts
    model_dir = os.path.dirname(model_file)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    data_dir = os.path.dirname(data_file)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    with open(model_file, "w") as f:
        f.write(model_code or "")
    with open(data_file, "w") as f:
        f.write(data_code or "")
    _notify(progress, f"Wrote files: {model_file} • {data_file}")

    # Final assessment if failed or if alignment disabled (like original behavior)
    if syntax_errors or not do_alignment:
        _notify(progress, "Requesting final assessment")
        assessment_prompt = _build_final_assessment_prompt(
            prompt, grammar_implementation, model_code or "", data_code or "", syntax_errors
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
        usage_totals["prompt_tokens"] += usage4.get("prompt_tokens", 0)
        usage_totals["completion_tokens"] += usage4.get("completion_tokens", 0)
        assessment_text = assessment_text_part or assessment_text

    _notify(progress, "Generation complete")

    # Pricing estimate
    try:
        from types import SimpleNamespace
    except Exception:
        SimpleNamespace = None  # type: ignore

    usage_summary = {
        "prompt_tokens": usage_totals.get("prompt_tokens", 0),
        "completion_tokens": usage_totals.get("completion_tokens", 0),
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
        # iterations reflects trials attempted (best-effort)
        return {
            "iterations": int(trials_run),
            "assessment": (assessment_text or "").strip(),
            "syntax_errors": syntax_errors,
            "cost": cost,
        }
    else:
        return (assessment_text or "").strip()


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


# ---------- Model discovery ----------


def list_openai_models(prefix: Optional[str] = "gpt") -> list[str]:
    return _base_list_openai_models(prefix=prefix)


def list_gemini_models(prefix: Optional[str] = "gemini") -> list[str]:
    return _base_list_gemini_models(prefix=prefix)


def list_ollama_models(prefix: Optional[str] = None) -> list[str]:
    return _base_list_ollama_models(prefix=prefix)


def list_models(llm_provider: Optional[str] = None, model_name: str = MODEL_NAME) -> list[str]:
    return _base_list_models(llm_provider=llm_provider, model_name=model_name)


def test():
    """
    Sanity test: list available models from all providers.
    """
    for provider in ("openai", "google", "ollama"):
        print(f"--- {provider.upper()} MODELS ---")
        models = list_models(provider)
        for m in models:
            print(f"• {m}")
        print()
