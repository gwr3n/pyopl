"""
GraphChain-based refactoring of pyopl_generative.

Nodes represent atomic operations (generate, validate, revise).
The DAG orchestrates iteration loops and conditional branching.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from ..pyopl_core import OPLCompiler, SemanticError
from .genai_pricing import estimate_costs as _estimate_costs
from .pyopl_generative import (
    FEW_SHOT_TOP_K,
    MAX_ITERATIONS,
    MAX_OUTPUT_TOKENS,
    Grammar,
    LLMProvider,
    _build_alignment_prompt,
    _build_final_assessment_prompt,
    _build_generation_prompt,
    _build_revision_prompt,
    _gather_few_shots,
    _get_grammar_implementation,
    _infer_provider,
    _json_loads_relaxed,
    _llm_generate_text,
    _notify,
)

logger = logging.getLogger(__name__)


# ============================================================================
# GraphChain Context & Execution Model
# ============================================================================


@dataclass
class ExecutionContext:
    """Shared state across all nodes in the GraphChain.

    Tracks input parameters, state evolution, validation results, and aggregated metrics
    throughout the generative solve workflow.
    """

    # Input parameters
    problem_prompt: str
    """The original problem description provided by the user."""

    model_file: str
    """Target output path for the generated .mod (PyOPL model) file."""

    data_file: str
    """Target output path for the generated .dat (data) file."""

    model_name: str
    """LLM model identifier (e.g., 'gpt-4', 'gemini-pro')."""

    grammar_mode: Grammar
    """Grammar mode for code generation (NONE, BNF, or CODE)."""

    provider: LLMProvider
    """LLM provider (OPENAI, GOOGLE, or OLLAMA)."""

    max_iterations: int
    """Maximum refinement iterations allowed."""

    do_alignment_check: bool
    """Whether to perform semantic alignment validation with problem prompt."""

    temperature: Optional[float]
    """LLM sampling temperature; None uses provider default."""

    stop: Optional[List[str]]
    """Stop tokens to terminate LLM generation."""

    progress: Optional[Callable[[str], None]]
    """Callback function for progress notifications."""

    few_shots: List[Dict[str, str]]
    """Few-shot examples for in-context learning (description, model, data)."""

    # Grammar and utility
    grammar_implementation: str = ""
    """Loaded grammar specification (GBNF, BNF, or code templates)."""

    # State: model/data
    model_code: str = ""
    """Current generated PyOPL model code."""

    data_code: str = ""
    """Current generated data specification."""

    # State: validation
    syntax_errors: List[str] = field(default_factory=list)
    """List of compilation/syntax errors encountered."""

    syntax_valid: bool = False
    """Whether the current code validates successfully."""

    aligned: bool = False
    """Whether the model/data aligns with the original problem prompt."""

    alignment_assessment: str = ""
    """LLM assessment of alignment or final model summary."""

    # State: iteration tracking
    iteration: int = 0
    """Refinement iteration count: number of revision cycles completed (set by GraphChainExecutor)."""

    total_prompt_tokens: int = 0
    """Cumulative prompt tokens used across all LLM calls."""

    total_completion_tokens: int = 0
    """Cumulative completion tokens used across all LLM calls."""

    # State: revision history
    last_revision_type: str = ""
    """Last applied revision type: 'syntax', 'alignment', or ''."""

    def __post_init__(self):
        """Validate ExecutionContext invariants."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if not self.model_file or not self.model_file.strip():
            raise ValueError("model_file must be a non-empty path")
        if not self.data_file or not self.data_file.strip():
            raise ValueError("data_file must be a non-empty path")


class NodeExecutionResult:
    """Result of executing a single node."""

    def __init__(self, context: ExecutionContext, success: bool = True, error: Optional[str] = None):
        self.context = context
        self.success = success
        self.error = error


# ============================================================================
# Node Base Class
# ============================================================================


class GraphNode(ABC):
    """Abstract base class for all GraphChain nodes."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        """Execute the node's logic and return updated context."""
        pass

    async def __call__(self, context: ExecutionContext) -> NodeExecutionResult:
        """Allow nodes to be called directly."""
        try:
            return await self.execute(context)
        except Exception as e:
            logger.exception(f"Node '{self.name}' failed: {e}")
            return NodeExecutionResult(context, success=False, error=str(e))


# ============================================================================
# Concrete Node Implementations
# ============================================================================


class GenerateNode(GraphNode):
    """Generate initial PyOPL model and data from the problem prompt."""

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Generating model and data from prompt")

        user_prompt = _build_generation_prompt(
            context.problem_prompt,
            context.grammar_implementation,
            few_shots=context.few_shots,
        )

        try:
            content, usage = _llm_generate_text(
                provider=context.provider,
                model_name=context.model_name,
                input_text=user_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=context.temperature,
                stop=context.stop,
                progress=context.progress,
                capture_usage=True,
            )

            if not content:
                return NodeExecutionResult(context, success=False, error="Empty generation response")

            result = _json_loads_relaxed(content)
            context.model_code = result.get("model", "")
            context.data_code = result.get("data", "")
            context.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            context.total_completion_tokens += usage.get("completion_tokens", 0) or 0

            _notify(context.progress, f"[{self.name}] Generated model and data")
            return NodeExecutionResult(context, success=True)

        except Exception as e:
            return NodeExecutionResult(context, success=False, error=f"Generation failed: {e}")


class CheckSyntaxNode(GraphNode):
    """Validate model and data syntax using OPLCompiler.

    Node execution always succeeds; check context.syntax_valid to determine validation result.
    """

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Validating syntax")
        compiler = OPLCompiler()
        context.syntax_errors = []
        context.syntax_valid = False

        try:
            compiler.compile_model(context.model_code, context.data_code)
            context.syntax_valid = True
            _notify(context.progress, f"[{self.name}] Syntax valid ✓")
            return NodeExecutionResult(context, success=True)
        except (SemanticError, Exception) as e:
            context.syntax_errors.append(str(e))
            error_msg = f"[{self.name}] Syntax error: {e}"
            _notify(context.progress, error_msg)
            logger.info(f"Validation produced {len(context.syntax_errors)} error(s); flagged for revision")
            logger.debug(f"Compilation failed: {e}", exc_info=True)
            # Node execution succeeded; syntax_valid flag reflects validation result
            return NodeExecutionResult(context, success=True)


class CheckAlignmentNode(GraphNode):
    """Assess semantic alignment with the original problem prompt."""

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Checking alignment with prompt")

        alignment_prompt = _build_alignment_prompt(
            context.problem_prompt,
            context.grammar_implementation,
            context.model_code,
            context.data_code,
        )

        try:
            content, usage = _llm_generate_text(
                provider=context.provider,
                model_name=context.model_name,
                input_text=alignment_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                stop=context.stop,
                progress=context.progress,
                capture_usage=True,
            )

            if not content:
                return NodeExecutionResult(context, success=False, error="Empty alignment response")

            result = _json_loads_relaxed(content)
            context.aligned = result.get("aligned", False)
            context.alignment_assessment = result.get("assessment", "").strip()
            context.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            context.total_completion_tokens += usage.get("completion_tokens", 0) or 0

            status = "✓" if context.aligned else "✗"
            _notify(
                context.progress,
                f"[{self.name}] Alignment: {status} {context.alignment_assessment[:60]}...",
            )
            return NodeExecutionResult(context, success=True)

        except Exception as e:
            return NodeExecutionResult(context, success=False, error=f"Alignment check failed: {e}")


class ReviseSyntaxNode(GraphNode):
    """Revise model/data to fix syntax/semantic errors."""

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Revising to fix {len(context.syntax_errors)} error(s)")

        revision_prompt = _build_revision_prompt(
            prompt=context.problem_prompt,
            grammar_implementation=context.grammar_implementation,
            model_code=context.model_code,
            data_code=context.data_code,
            compile_errors=context.syntax_errors,
            alignment_assessment=None,
            few_shots=context.few_shots,
        )

        try:
            content, usage = _llm_generate_text(
                provider=context.provider,
                model_name=context.model_name,
                input_text=revision_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=context.temperature,
                stop=context.stop,
                progress=context.progress,
                capture_usage=True,
            )

            if not content:
                return NodeExecutionResult(context, success=False, error="Empty revision response")

            result = _json_loads_relaxed(content)
            context.model_code = result.get("model", context.model_code)
            context.data_code = result.get("data", context.data_code)
            context.last_revision_type = "syntax"
            context.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            context.total_completion_tokens += usage.get("completion_tokens", 0) or 0

            _notify(context.progress, f"[{self.name}] Model revised (syntax fix)")
            return NodeExecutionResult(context, success=True)

        except Exception as e:
            return NodeExecutionResult(context, success=False, error=f"Syntax revision failed: {e}")


class ReviseAlignmentNode(GraphNode):
    """Revise model/data to improve alignment with the problem."""

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Revising to improve alignment")

        revision_prompt = _build_revision_prompt(
            prompt=context.problem_prompt,
            grammar_implementation=context.grammar_implementation,
            model_code=context.model_code,
            data_code=context.data_code,
            compile_errors=None,
            alignment_assessment=context.alignment_assessment,
            few_shots=context.few_shots,
        )

        try:
            content, usage = _llm_generate_text(
                provider=context.provider,
                model_name=context.model_name,
                input_text=revision_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=context.temperature,
                stop=context.stop,
                progress=context.progress,
                capture_usage=True,
            )

            if not content:
                return NodeExecutionResult(context, success=False, error="Empty revision response")

            result = _json_loads_relaxed(content)
            context.model_code = result.get("model", context.model_code)
            context.data_code = result.get("data", context.data_code)
            context.last_revision_type = "alignment"
            context.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            context.total_completion_tokens += usage.get("completion_tokens", 0) or 0

            _notify(context.progress, f"[{self.name}] Model revised (alignment fix)")
            return NodeExecutionResult(context, success=True)

        except Exception as e:
            return NodeExecutionResult(context, success=False, error=f"Alignment revision failed: {e}")


class FinalAssessmentNode(GraphNode):
    """Generate final assessment of model/data alignment and correctness."""

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Generating final assessment")

        assessment_prompt = _build_final_assessment_prompt(
            context.problem_prompt,
            context.grammar_implementation,
            context.model_code,
            context.data_code,
            context.syntax_errors,
        )

        try:
            content, usage = _llm_generate_text(
                provider=context.provider,
                model_name=context.model_name,
                input_text=assessment_prompt,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                stop=context.stop,
                progress=context.progress,
                capture_usage=True,
            )

            if not content:
                content = "Unable to generate assessment."

            context.alignment_assessment = content.strip()
            context.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            context.total_completion_tokens += usage.get("completion_tokens", 0) or 0

            _notify(context.progress, f"[{self.name}] Assessment complete")
            return NodeExecutionResult(context, success=True)

        except Exception as e:
            return NodeExecutionResult(context, success=False, error=f"Final assessment failed: {e}")


class SaveFilesNode(GraphNode):
    """Write model and data to disk with atomic file operations."""

    async def execute(self, context: ExecutionContext) -> NodeExecutionResult:
        _notify(context.progress, f"[{self.name}] Writing files to disk")

        try:
            import tempfile

            # Ensure directories exist
            model_dir = os.path.dirname(context.model_file) or "."
            data_dir = os.path.dirname(context.data_file) or "."

            if model_dir != ".":
                os.makedirs(model_dir, exist_ok=True)
            if data_dir != ".":
                os.makedirs(data_dir, exist_ok=True)

            # Write model file atomically (temp -> rename)
            model_fd, model_temp = tempfile.mkstemp(dir=model_dir, prefix=".tmp_", suffix=".mod.tmp")
            try:
                with os.fdopen(model_fd, "w") as f:
                    f.write(context.model_code)
                os.replace(model_temp, context.model_file)
            except Exception:
                try:
                    os.close(model_fd)
                    os.unlink(model_temp)
                except Exception:
                    pass
                raise

            # Write data file atomically (temp -> rename)
            data_fd, data_temp = tempfile.mkstemp(dir=data_dir, prefix=".tmp_", suffix=".dat.tmp")
            try:
                with os.fdopen(data_fd, "w") as f:
                    f.write(context.data_code)
                os.replace(data_temp, context.data_file)
            except Exception:
                try:
                    os.close(data_fd)
                    os.unlink(data_temp)
                except Exception:
                    pass
                raise

            _notify(context.progress, f"[{self.name}] Files written: {context.model_file} • {context.data_file}")
            return NodeExecutionResult(context, success=True)

        except Exception as e:
            return NodeExecutionResult(context, success=False, error=f"File write failed: {e}")


# ============================================================================
# GraphChain Orchestrator
# ============================================================================


class GraphChainExecutor:
    """DAG executor for PyOPL generative workflow.

    Nodes are stateless and instantiated fresh per execution for clarity and simplicity.
    Manages iteration loop control and conditional branching explicitly in execute().

    Workflow:

        Generate → CheckSyntax ─→[pass]→ CheckAlignment ─→[pass]→ SaveFiles
                       ↓                       ↓
                     [fail]                [fail]
                       ↓                       ↓
                   ReviseSyntax ───────────────┘
                        ↑
                        └─ (loop back to CheckSyntax if iterations < max)

    Iteration Control:
    - Starts: Generate once, then refinement_iteration = 1
    - Loop: CheckSyntax → [invalid] → ReviseSyntax → CheckSyntax (repeat while refinement_iteration < max_iterations)
    - Then: CheckAlignment → [misaligned] → ReviseAlignment → CheckSyntax (repeat while refinement_iteration < max_iterations)
    - Exit: syntax_valid AND (aligned OR alignment_check disabled), OR refinement_iteration >= max_iterations
    - Optional: FinalAssessment (if unresolved errors or alignment disabled)
    - Always: SaveFiles (final step)
    """

    def __init__(self, max_iterations: int = MAX_ITERATIONS):
        self.max_iterations = max_iterations

        # Instantiate nodes
        self.gen = GenerateNode("generate")
        self.check_syntax = CheckSyntaxNode("check_syntax")
        self.check_alignment = CheckAlignmentNode("check_alignment")
        self.revise_syntax = ReviseSyntaxNode("revise_syntax")
        self.revise_alignment = ReviseAlignmentNode("revise_alignment")
        self.final_assessment = FinalAssessmentNode("final_assessment")
        self.save_files = SaveFilesNode("save_files")

    async def execute(self, context: ExecutionContext) -> ExecutionContext:
        """
        Execute the DAG:

        1. Generate model/data
        2. Check syntax
           - If invalid: revise syntax → check syntax (loop if iterations < max)
           - If valid: check alignment
        3. Check alignment (if enabled)
           - If aligned: save & done
           - If not aligned: revise alignment → check syntax (loop if iterations < max)
        4. If final iteration or max iterations reached: final assessment
        5. Save files
        """

        # Initialize grammar implementation
        context.grammar_implementation = _get_grammar_implementation(context.grammar_mode)

        # Step 1: Generate
        result = await self.gen(context)
        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")
        context = result.context

        # Step 2+: Iterative refinement loop
        # Iteration counter is managed here for explicit control flow
        refinement_iteration = 1
        while refinement_iteration < context.max_iterations:
            # Check syntax
            result = await self.check_syntax(context)
            if not result.success:
                raise RuntimeError(f"Syntax check failed: {result.error}")
            context = result.context

            # Branch A: Syntax invalid → revise & loop
            if not context.syntax_valid:
                _notify(context.progress, f"Iteration {refinement_iteration}/{context.max_iterations}: syntax errors found")
                result = await self.revise_syntax(context)
                if not result.success:
                    raise RuntimeError(f"Syntax revision failed: {result.error}")
                context = result.context
                refinement_iteration += 1
                continue  # Loop back to syntax check

            # Branch B: Syntax valid, but alignment check disabled → exit loop
            if not context.do_alignment_check:
                _notify(context.progress, "Syntax valid; alignment check disabled. Stopping.")
                break

            # Branch C: Syntax valid, check alignment
            result = await self.check_alignment(context)
            if not result.success:
                raise RuntimeError(f"Alignment check failed: {result.error}")
            context = result.context

            # If aligned: exit loop
            if context.aligned:
                _notify(context.progress, "Aligned ✓ Stopping.")
                break

            # If not aligned: revise & loop
            _notify(context.progress, f"Iteration {refinement_iteration}/{context.max_iterations}: misaligned, revising")
            result = await self.revise_alignment(context)
            if not result.success:
                raise RuntimeError(f"Alignment revision failed: {result.error}")
            context = result.context
            refinement_iteration += 1

        # Track final iteration count for statistics
        context.iteration = refinement_iteration - 1

        # Step 3: Final assessment (if needed)
        # Run final assessment if:
        # - There are unresolved syntax errors to explain, OR
        # - Alignment check was disabled (no validation performed)
        # Skip if syntax is valid AND alignment succeeded
        if context.syntax_errors or not context.do_alignment_check:
            result = await self.final_assessment(context)
            if not result.success:
                logger.warning(f"Final assessment failed (non-fatal): {result.error}")
            context = result.context

        # Step 4: Save files
        result = await self.save_files(context)
        if not result.success:
            raise RuntimeError(f"File save failed: {result.error}")
        context = result.context

        return context


# ============================================================================
# Public API (Async)
# ============================================================================


async def generative_solve_async(
    prompt: str,
    model_file: str,
    data_file: str,
    model_name: str = "gpt-5",
    mode: Grammar = Grammar.BNF,
    iterations: int = MAX_ITERATIONS,
    return_statistics: bool = False,
    alignment_check: Optional[bool] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    llm_provider: Optional[str] = "openai",
    progress: Optional[Callable[[str], None]] = None,
    few_shot: bool = True,
) -> Union[str, Dict[str, Any]]:
    """
    Async version of generative_solve using GraphChain orchestration.

    Same signature and behavior as the original generative_solve,
    but uses explicit node-based DAG execution.
    """

    iterations = max(1, int(iterations))
    do_alignment = True if alignment_check is None else bool(alignment_check)
    provider = _infer_provider(llm_provider, model_name)

    _notify(
        progress,
        f"GraphChain: provider={provider.value} model={model_name} iterations={iterations} "
        f"alignment={'on' if do_alignment else 'off'}",
    )

    # Gather few-shot examples
    few_shots: List[Dict[str, str]] = (
        _gather_few_shots(prompt, k=FEW_SHOT_TOP_K, models_dir=None, progress=progress) if few_shot else []
    )

    # Initialize execution context
    context = ExecutionContext(
        problem_prompt=prompt,
        model_file=model_file,
        data_file=data_file,
        model_name=model_name,
        grammar_mode=mode,
        provider=provider,
        max_iterations=iterations,
        do_alignment_check=do_alignment,
        temperature=temperature,
        stop=stop,
        progress=progress,
        few_shots=few_shots,
    )

    # Execute GraphChain
    executor = GraphChainExecutor(max_iterations=iterations)
    try:
        context = await executor.execute(context)
    except Exception as e:
        logger.exception(f"GraphChain execution failed: {e}")
        raise RuntimeError(f"Generative solve failed: {e}") from e

    # Estimate costs
    estimated_costs: Dict[str, Any] = {}
    try:
        from types import SimpleNamespace

        args = SimpleNamespace(model=model_name)
        usage_dict = {
            "prompt_tokens": context.total_prompt_tokens,
            "completion_tokens": context.total_completion_tokens,
        }
        estimated_costs = _estimate_costs(args, usage_dict) or {}
    except Exception:
        pass

    cost = {
        "model": model_name,
        "usage": {
            "prompt_tokens": context.total_prompt_tokens,
            "completion_tokens": context.total_completion_tokens,
        },
        "estimated_costs": estimated_costs,
    }

    _notify(progress, f"[GraphChain] Complete. Cost: {cost}")
    logger.info(
        f"GraphChain execution complete: "
        f"iterations={context.iteration}, "
        f"syntax_valid={context.syntax_valid}, "
        f"tokens={context.total_prompt_tokens + context.total_completion_tokens}"
    )

    if return_statistics:
        return {
            "iterations": context.iteration,
            "assessment": context.alignment_assessment.strip(),
            "syntax_errors": context.syntax_errors,
            "cost": cost,
        }
    else:
        return context.alignment_assessment.strip()


# ============================================================================
# Sync Wrapper (for backwards compatibility)
# ============================================================================


def generative_solve_graphchain(
    prompt: str,
    model_file: str,
    data_file: str,
    model_name: str = "gpt-5",
    mode: Grammar = Grammar.BNF,
    iterations: int = MAX_ITERATIONS,
    return_statistics: bool = False,
    alignment_check: Optional[bool] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    llm_provider: Optional[str] = "openai",
    progress: Optional[Callable[[str], None]] = None,
    few_shot: bool = True,
) -> Union[str, Dict[str, Any]]:
    """
    Synchronous wrapper for the async GraphChain executor.
    Uses asyncio.run() to execute the async version.
    """
    import asyncio
    import warnings

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Not in async context; safe to use asyncio.run()
        return asyncio.run(
            generative_solve_async(
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
        )

    # If we reach here, we're already in an async context
    warnings.warn("generative_solve_graphchain called from async context; use generative_solve_async instead")
    raise RuntimeError("Cannot use sync wrapper from async context")
