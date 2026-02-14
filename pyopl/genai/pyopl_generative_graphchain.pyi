"""Type stubs for pyopl_generative_graphchain module."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Union, overload

from .pyopl_generative import Grammar, LLMProvider

@dataclass
class ExecutionContext:
    """Shared state across all nodes in the GraphChain."""

    problem_prompt: str
    model_file: str
    data_file: str
    model_name: str
    grammar_mode: Grammar
    provider: LLMProvider
    max_iterations: int
    do_alignment_check: bool
    temperature: Optional[float]
    stop: Optional[List[str]]
    progress: Optional[Callable[[str], None]]
    few_shots: List[Dict[str, str]]
    grammar_implementation: str
    model_code: str
    data_code: str
    syntax_errors: List[str]
    syntax_valid: bool
    aligned: bool
    alignment_assessment: str
    iteration: int
    total_prompt_tokens: int
    total_completion_tokens: int
    last_revision_type: str

    def __post_init__(self) -> None: ...

class NodeExecutionResult:
    """Result of executing a single node."""

    context: ExecutionContext
    success: bool
    error: Optional[str]

    def __init__(
        self,
        context: ExecutionContext,
        success: bool = ...,
        error: Optional[str] = ...,
    ) -> None: ...

class GraphChainExecutor:
    """DAG executor for PyOPL generative workflow."""

    max_iterations: int

    def __init__(self, max_iterations: int = ...) -> None: ...
    async def execute(self, context: ExecutionContext) -> ExecutionContext: ...

@overload
async def generative_solve_async(
    prompt: str,
    model_file: str,
    data_file: str,
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
async def generative_solve_async(
    prompt: str,
    model_file: str,
    data_file: str,
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
@overload
async def generative_solve_async(
    prompt: str,
    model_file: str,
    data_file: str,
    model_name: str = ...,
    mode: Grammar = ...,
    iterations: int = ...,
    return_statistics: bool = ...,
    alignment_check: Optional[bool] = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    llm_provider: Optional[str] = ...,
    progress: Optional[Callable[[str], None]] = ...,
    few_shot: bool = ...,
) -> Union[str, Dict[str, Any]]: ...
@overload
def generative_solve_graphchain(
    prompt: str,
    model_file: str,
    data_file: str,
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
def generative_solve_graphchain(
    prompt: str,
    model_file: str,
    data_file: str,
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
@overload
def generative_solve_graphchain(
    prompt: str,
    model_file: str,
    data_file: str,
    model_name: str = ...,
    mode: Grammar = ...,
    iterations: int = ...,
    return_statistics: bool = ...,
    alignment_check: Optional[bool] = ...,
    temperature: Optional[float] = ...,
    stop: Optional[List[str]] = ...,
    llm_provider: Optional[str] = ...,
    progress: Optional[Callable[[str], None]] = ...,
    few_shot: bool = ...,
) -> Union[str, Dict[str, Any]]: ...
