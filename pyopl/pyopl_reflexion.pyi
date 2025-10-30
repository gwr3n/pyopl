from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, overload

MAX_ITERATIONS: int
MAX_OUTPUT_TOKENS: Optional[int]
LLM_PROVIDER: str
MODEL_NAME: str
ALIGNMENT_CHECK: bool

class LLMProvider(Enum):
    OPENAI = ...
    GOOGLE = ...
    OLLAMA = ...

class Grammar(Enum):
    NONE = ...
    BNF = ...
    CODE = ...

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
