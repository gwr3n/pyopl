"""FastMCP server exposing Rhetor functionality as MCP tools.

This module can be run as a standalone server or integrated into an existing
MCP setup. It exposes tools for:

- listing available LLM models and generative methods
- generating model/data files from natural-language prompts
- requesting feedback on an existing model/data pair
- producing an end-to-end "insight" workflow: generate, solve, summarize

Example VS Code MCP config (.vscode/mcp.json):

{
    "servers": {
        "Rhetor MCP": {
            "type": "stdio",
            "command": "${workspaceFolder}/venv/bin/python",
            "args": ["-m", "pyopl.rhetor_mcp"]
        }
    },
    "inputs": []
}
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, TypeVar, Union

from mcp.server.fastmcp import FastMCP

from . import generative_feedback, generative_solve, solve
from .genai._strategy_base import (
    LLMProvider,
    list_gemini_models,
    list_ollama_models,
    list_openai_models,
)

PathLike = Union[str, Path]
T = TypeVar("T")

DEFAULT_SOLVER = "highs"
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5"

METHODS: list[tuple[str, str]] = [
    ("SyntAGM", "pyopl_generative"),
    ("Standard", "pyopl_standard"),
    ("Chain of Thought", "pyopl_chain_of_thought"),
    ("Tree of Thoughts", "pyopl_tree_of_thoughts"),
    ("CAFA", "pyopl_cafa"),
    ("Chain of Experts", "pyopl_chain_of_experts"),
    ("Reflexion", "pyopl_reflexion"),
]

mcp = FastMCP("Rhetor MCP")


def _normalize_solver(solver: Optional[str]) -> str:
    """Normalize solver names for compiler/backend compatibility."""
    if not solver:
        return "scipy"

    normalized = solver.strip().lower()

    solver_aliases = {
        "highs": "scipy",
        "scipy": "scipy",
        "gurobi": "gurobi",
    }
    return solver_aliases.get(normalized, normalized)


def _solve_backend(solver: Optional[str]) -> str:
    """Normalize solver names for solve() backend selection."""
    return "gurobi" if _normalize_solver(solver) == "gurobi" else "scipy"


def _normalize_provider(provider: Optional[str]) -> str:
    """Normalize supported LLM provider names."""
    if not provider:
        return DEFAULT_PROVIDER

    normalized = provider.strip().lower()
    provider_aliases = {
        "openai": "openai",
        "google": "google",
        "gemini": "google",
        "ollama": "ollama",
    }

    if normalized not in provider_aliases:
        raise ValueError(f"Unsupported provider '{provider}'. " f"Expected one of: {', '.join(sorted(provider_aliases))}.")

    return provider_aliases[normalized]


def _build_llm_kwargs(
    llm_model: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, str]:
    """Build kwargs for generative helper calls."""
    kwargs: dict[str, str] = {}
    if llm_model:
        kwargs["model_name"] = llm_model
    if provider:
        kwargs["llm_provider"] = _normalize_provider(provider)
    return kwargs


def _run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous callable in a dedicated worker thread."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(partial(func, *args, **kwargs))
        return future.result()


def _run_coro_in_thread(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async coroutine safely from a sync context in a worker thread."""

    def _runner() -> T:
        return asyncio.run(coro_factory())

    return _run_in_thread(_runner)


def _try_import_graphchain():
    """Try to import async GraphChain generation implementation."""
    try:
        from .genai.pyopl_generative_graphchain import generative_solve_async

        return generative_solve_async
    except Exception:
        return None


def _generate_with_best_available_backend(
    prompt: str,
    model_file: str,
    data_file: str,
    *,
    iterations: int = 5,
    llm_model: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict:
    """Use async GraphChain if available, otherwise fall back to sync generation."""
    llm_kwargs = _build_llm_kwargs(llm_model=llm_model, provider=provider)
    generative_solve_async = _try_import_graphchain()

    if generative_solve_async is not None:

        async def _coro() -> dict:
            return await generative_solve_async(
                prompt,
                model_file,
                data_file,
                model_name=llm_kwargs.get("model_name", DEFAULT_MODEL),
                iterations=iterations,
                return_statistics=True,
                llm_provider=llm_kwargs.get("llm_provider", DEFAULT_PROVIDER),
            )

        return _run_coro_in_thread(_coro)

    return _run_in_thread(
        generative_solve,
        prompt,
        model_file,
        data_file,
        iterations=iterations,
        return_statistics=True,
        **llm_kwargs,
    )


def _ask_for_feedback(
    prompt: str,
    model_file: str,
    data_file: str,
    *,
    llm_model: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict | str:
    """Get feedback on an existing model/data pair."""
    llm_kwargs = _build_llm_kwargs(llm_model=llm_model, provider=provider)
    return _run_in_thread(
        generative_feedback,
        prompt,
        model_file,
        data_file,
        **llm_kwargs,
    )


def list_providers() -> list:
    """List supported LLM providers."""
    # Return canonical provider names from the LLMProvider enum
    try:
        return [provider.value for provider in LLMProvider]
    except Exception:
        # Fallback for safety in environments where the enum isn't available
        return ["openai", "google", "ollama"]


def list_models(
    provider: Optional[str] = DEFAULT_PROVIDER,
    prefix: Optional[str] = None,
) -> list:
    """List available models for a given provider."""
    normalized_provider = _normalize_provider(provider)

    if normalized_provider == "openai":
        return list_openai_models(prefix=prefix) if prefix else list_openai_models()
    if normalized_provider == "google":
        return list_gemini_models(prefix=prefix) if prefix else list_gemini_models()
    if normalized_provider == "ollama":
        return list_ollama_models(prefix=prefix) if prefix else list_ollama_models()

    raise ValueError(f"Unsupported provider: {provider}")


@mcp.tool()
def list_providers_tool() -> list:
    """Return supported LLM provider identifiers.

    Returns a list of canonical provider names (strings), e.g.
    `['openai', 'google', 'ollama']`.

    Returns:
        A list of provider name strings.
    """
    return list_providers()


@mcp.tool()
def list_models_tool(
    provider: Optional[str] = DEFAULT_PROVIDER,
    prefix: Optional[str] = None,
) -> list:
    """List available LLM models for a provider.

    The `provider` argument is normalized (aliases like ``gemini`` ->
    ``google`` are accepted). When `prefix` is supplied, the returned
    models are filtered to those starting with the prefix.

    Args:
        provider: Provider identifier (e.g., ``openai``, ``google``, ``ollama``).
        prefix: Optional prefix for filtering model names.

    Returns:
        A list of model names (strings). The exact format depends on the
        provider but is typically a list of model identifier strings.

    Raises:
        ValueError: if an unsupported provider is supplied.
    """
    return list_models(provider=provider, prefix=prefix)


@mcp.tool()
def list_methods_tool() -> list[tuple[str, str]]:
    """List available generative method choices.

    Returns a list of tuples ``(display_name, method_id)`` where
    ``display_name`` is a human-friendly label and ``method_id`` is the
    identifier used by the MCP client to select the method.
    """
    return METHODS


@mcp.tool()
def generate_tool(
    prompt: str,
    model_file: str,
    data_file: str,
    llm_model: Optional[str] = None,
    provider: Optional[str] = None,
    iterations: int = 5,
) -> dict:
    """Generate an OPL model and data file from a natural-language prompt.

    This is a potentially long-running operation that uses a generative
    LLM to produce a `.mod` model file and a `.dat` data file at the
    specified output paths. The function writes those files as a side
    effect and returns statistics and metadata describing the
    generation/refinement process.

    Important: generative backends require appropriate environment
    variables (for example `OPENAI_API_KEY` or `GEMINI_API_KEY`) to be set.

    Args:
        prompt: Natural-language problem description.
        model_file: Path where the generated `.mod` will be written.
        data_file: Path where the generated `.dat` will be written.
        llm_model: Optional explicit model identifier to use.
        provider: Optional LLM provider override.
        iterations: Number of refinement iterations to run.

    Returns:
        A dictionary containing generation statistics and metadata. Keys
        typically include timing, chosen model, and any per-iteration
        diagnostics; exact contents are backend-dependent.

    Raises:
        Exceptions raised by the LLM client or I/O operations are
        propagated to the caller.
    """
    return _generate_with_best_available_backend(
        prompt,
        model_file,
        data_file,
        iterations=iterations,
        llm_model=llm_model,
        provider=provider,
    )


@mcp.tool()
def ask_tool(
    prompt: str,
    model_file: str,
    data_file: str,
    llm_model: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict | str:
    """Ask an LLM for feedback on an existing OPL model and data file.

    The tool reads the model and data from the supplied filesystem paths
    and sends a feedback request to the configured LLM provider. The
    returned value may be a plain string (text feedback) or a structured
    dictionary containing richer metadata (for example, score, suggested
    edits, or structured comments) depending on the backend.

    Args:
        prompt: Question or feedback request about the supplied model/data pair.
        model_file: Path to an existing `.mod` file (read by the tool).
        data_file: Path to an existing `.dat` file (read by the tool).
        llm_model: Optional model name to use.
        provider: Optional LLM provider.

    Returns:
        A feedback string or a structured dictionary, backend-dependent.

    Raises:
        Exceptions from file I/O or the LLM client are propagated.
    """
    return _ask_for_feedback(
        prompt,
        model_file,
        data_file,
        llm_model=llm_model,
        provider=provider,
    )


@mcp.tool()
def insight_tool(
    prompt: str,
    provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    iterations: int = 5,
    solver: str = DEFAULT_SOLVER,
) -> dict:
    """Generate, solve, and provide insights on an optimization problem from a prompt.

    This tool creates a model/data pair in a persistent temporary directory,
    attempts to solve the generated problem, and produces a non-technical summary
    of the results.

    Returns:
        A dictionary containing:
            - model_path: Path to the generated `.mod` file.
            - data_path: Path to the generated `.dat` file.
            - stats: Generation statistics.
            - results: Solver output or an error payload.
            - feedback: LLM-generated explanation or an error payload.
            - markdown: A human-readable markdown summary.

        Retention and cleanup:
                - Artifacts are written into a persistent directory under the system
                    temporary directory (see `tempfile.gettempdir()`) and are named with
                    the prefix ``pyopl_mcp_``. Example on macOS: ``/var/folders/.../T/pyopl_mcp_*``.
                - The directories and files are intentionally persistent and are NOT
                    removed automatically by this function so the MCP client can access
                    the produced `.mod`/`.dat` files after the call returns.
                - The returned ``model_path`` and ``data_path`` point to the exact
                    files; callers are responsible for removing the artifact directory
                    when it is no longer needed (either manually or via an explicit
                    cleanup tool).
    """
    # NOTE:
    # We intentionally use a persistent temp directory here because the returned
    # model_path/data_path should remain valid after this function returns.
    artifact_dir = Path(tempfile.mkdtemp(prefix="pyopl_mcp_"))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"gen_pyopl_{timestamp}"

    model_path = artifact_dir / f"{base_name}.mod"
    data_path = artifact_dir / f"{base_name}.dat"

    stats = _generate_with_best_available_backend(
        prompt,
        str(model_path),
        str(data_path),
        iterations=iterations,
        llm_model=llm_model,
        provider=provider,
    )

    try:
        results = solve(
            str(model_path),
            str(data_path),
            solver=_solve_backend(solver),
        )
    except Exception as exc:
        results = {"error": f"Error solving generated model: {exc}"}

    solution_json = json.dumps(results, indent=2, sort_keys=True, default=str)
    feedback_prompt = (
        "Translate the following optimization solution into clear, "
        "non-technical language for a lay user. Include key findings and "
        "suggested next steps.\n\n"
        f"Solution:\n{solution_json}"
    )

    try:
        feedback = _ask_for_feedback(
            feedback_prompt,
            str(model_path),
            str(data_path),
            llm_model=llm_model,
            provider=provider,
        )
    except Exception as exc:
        feedback = {"error": f"Error generating feedback: {exc}"}

    if isinstance(feedback, dict):
        summary = feedback.get("feedback") or feedback.get("summary") or json.dumps(feedback, indent=2, default=str)
    else:
        summary = str(feedback)

    markdown = "\n".join(
        [
            "# GenAI Insight",
            "",
            "## Problem Description",
            "",
            prompt,
            "",
            "## Insight",
            "",
            summary,
            "",
        ]
    )

    return {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "stats": stats,
        "results": results,
        "feedback": feedback,
        "markdown": markdown,
    }


if __name__ == "__main__":
    mcp.run()
