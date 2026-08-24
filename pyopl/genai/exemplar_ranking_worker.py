from __future__ import annotations

import multiprocessing
import queue
from pathlib import Path
from typing import Any, Optional

from .rag_helper import DEFAULT_EMBEDDING_MODEL


def _run_ranking_worker(
    models_dirs: list[str],
    command_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        from .rag_helper import _load_model, rank_problem_descriptions

        if _load_model(DEFAULT_EMBEDDING_MODEL) is None:
            result_queue.put(("startup_error", None, "Embedding model could not be loaded."))
            return
        result_queue.put(("ready", None, None))
        while True:
            command, generation, query = command_queue.get()
            if command == "stop":
                return
            if command != "rank":
                continue
            try:
                hits = rank_problem_descriptions(query=query, models_dir=models_dirs, top_k=0)
                result_queue.put(("success", generation, hits))
            except Exception as exc:
                result_queue.put(("error", generation, f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        result_queue.put(("startup_error", None, f"{type(exc).__name__}: {exc}"))


class ExemplarRankingWorker:
    """Own a reusable embedding-ranking process for one retrieval dialog."""

    def __init__(self, models_dirs: list[Path]) -> None:
        self._models_dirs = [str(path) for path in models_dirs]
        self._commands: Optional[multiprocessing.Queue] = None
        self._results: Optional[multiprocessing.Queue] = None
        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        if self._process is not None:
            return
        self._commands = multiprocessing.Queue()
        self._results = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_run_ranking_worker,
            args=(self._models_dirs, self._commands, self._results),
        )
        self._process.start()

    def submit(self, generation: int, query: str) -> None:
        if self._commands is None:
            raise RuntimeError("Exemplar ranking worker has not been started.")
        self._commands.put(("rank", generation, query))

    def get_nowait(self) -> tuple[str, Optional[int], Any]:
        if self._results is None:
            raise queue.Empty
        return self._results.get_nowait()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def close(self) -> None:
        process = self._process
        commands = self._commands
        results = self._results
        self._process = None
        self._commands = None
        self._results = None

        if process is not None:
            try:
                if process.is_alive() and commands is not None:
                    commands.put(("stop", None, None))
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
            except Exception:
                pass
        for ipc_queue in (commands, results):
            if ipc_queue is None:
                continue
            try:
                ipc_queue.close()
                ipc_queue.cancel_join_thread()
            except Exception:
                pass