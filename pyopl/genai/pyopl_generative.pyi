from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, overload

MAX_ITERATIONS: int
MAX_OUTPUT_TOKENS: Optional[int]
LLM_PROVIDER: str
MODEL_NAME: str
ALIGNMENT_CHECK: bool
FEW_SHOT_TOP_K: int
FEW_SHOT_MAX_CHARS: int

class LLMProvider(Enum):
    OPENAI = ...
    GOOGLE = ...
    OLLAMA = ...

class Grammar(Enum):
    NONE = ...
    BNF = ...
    CODE = ...

# Private utility functions
def _notify(progress: Optional[Callable[[str], None]], msg: str) -> None: ...
def _get_grammar_implementation(mode: Grammar) -> str: ...
def _gather_few_shots(
    problem_description: str,
    k: int = ...,
    models_dir: Optional[str] = ...,
    progress: Optional[Callable[[str], None]] = ...,
) -> List[Dict[str, str]]: ...
def _json_loads_relaxed(text: str) -> Dict[str, Any]: ...
def _infer_provider(llm_provider: Optional[str], model_name: str) -> LLMProvider: ...
def _build_generation_prompt(
    prompt: str,
    grammar_implementation: str,
    few_shots: Optional[List[Dict[str, str]]] = ...,
) -> str: ...
def _build_alignment_prompt(
    prompt: str,
    grammar_implementation: str,
    model_code: str,
    data_code: str,
) -> str: ...
def _build_revision_prompt(
    prompt: str,
    grammar_implementation: str,
    model_code: str,
    data_code: str,
    compile_errors: Optional[List[str]] = ...,
    alignment_assessment: Optional[str] = ...,
    few_shots: Optional[List[Dict[str, str]]] = ...,
) -> str: ...
def _build_final_assessment_prompt(
    prompt: str,
    grammar_implementation: str,
    model_code: str,
    data_code: str,
    syntax_errors: Optional[List[str]] = ...,
) -> str: ...
@overload
def _ollama_generate_text(
    model_name: str,
    prompt: str,
    num_predict: Optional[int] = ...,
    return_usage: Literal[True] = ...,
) -> Tuple[str, Dict[str, int]]: ...
@overload
def _ollama_generate_text(
    model_name: str,
    prompt: str,
    num_predict: Optional[int] = ...,
    return_usage: Literal[False] = ...,
) -> str: ...
@overload
def _llm_generate_text(
    provider: LLMProvider,
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    progress: Optional[Callable[[str], None]] = ...,
    capture_usage: Literal[True] = ...,
) -> Tuple[str, Dict[str, int]]: ...
@overload
def _llm_generate_text(
    provider: LLMProvider,
    model_name: str,
    input_text: str,
    max_tokens: Optional[int] = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    progress: Optional[Callable[[str], None]] = ...,
    capture_usage: Literal[False] = ...,
) -> str: ...
@overload
def generative_solve(
    prompt,
    model_file,
    data_file,
    model_name: str = ...,
    mode: Grammar = ...,
    iterations: int = ...,
    return_statistics: Literal[True] = ...,
    alignment_check: Optional[bool] = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    llm_provider: Optional[str] = ...,
    progress: Optional[Callable[[str], None]] = ...,
    few_shot: bool = ...,
    use_graphchain: bool = ...,
) -> Dict[str, Any]: ...
@overload
def generative_solve(
    prompt,
    model_file,
    data_file,
    model_name: str = ...,
    mode: Grammar = ...,
    iterations: int = ...,
    return_statistics: Literal[False] = ...,
    alignment_check: Optional[bool] = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    llm_provider: Optional[str] = ...,
    progress: Optional[Callable[[str], None]] = ...,
    few_shot: bool = ...,
    use_graphchain: bool = ...,
) -> str: ...
def generative_feedback(
    prompt,
    model_file,
    data_file,
    model_name: str = ...,
    mode: Grammar = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    llm_provider: Optional[str] = ...,
    progress: Optional[Callable[[str], None]] = ...,
) -> Dict[str, str]: ...
def list_models(llm_provider: Optional[str] = ..., model_name: str = ...) -> List[str]: ...
def list_openai_models(prefix: Optional[str] = "gpt") -> List[str]: ...
def list_gemini_models(prefix: Optional[str] = "gemini") -> List[str]: ...
def list_ollama_models(prefix: Optional[str] = ...) -> List[str]: ...
