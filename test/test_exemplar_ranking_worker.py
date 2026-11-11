import unittest
from pathlib import Path
from queue import Empty
from unittest.mock import MagicMock, patch

from pyopl.genai import exemplar_ranking_worker
from pyopl.genai.exemplar_ranking_worker import ExemplarRankingWorker


class TestExemplarRankingWorker(unittest.TestCase):
    def test_worker_must_be_started_before_submission(self):
        worker = ExemplarRankingWorker([])

        with self.assertRaisesRegex(RuntimeError, "has not been started"):
            worker.submit(1, "query")
        with self.assertRaises(Empty):
            worker.get_nowait()
        self.assertFalse(worker.is_alive())
        worker.close()

    def test_start_is_idempotent_and_submit_queues_generation(self):
        worker = ExemplarRankingWorker([Path("models")])
        commands = MagicMock()
        results = MagicMock()

        with (
            patch.object(exemplar_ranking_worker.multiprocessing, "Queue", side_effect=[commands, results]),
            patch.object(exemplar_ranking_worker.multiprocessing, "Process") as process_class,
        ):
            worker.start()
            worker.start()
            worker.submit(7, "query")
            results.get_nowait.return_value = ("success", 7, [])
            self.assertEqual(worker.get_nowait(), ("success", 7, []))

        process_class.assert_called_once_with(
            target=exemplar_ranking_worker._run_ranking_worker,
            args=(["models"], commands, results),
        )
        process_class.return_value.start.assert_called_once_with()
        commands.put.assert_called_once_with(("rank", 7, "query"))

    def test_worker_reports_unavailable_embedding_model(self):
        commands = MagicMock()
        results = MagicMock()

        with patch("pyopl.genai.rag_helper._load_model", return_value=None):
            exemplar_ranking_worker._run_ranking_worker(["models"], commands, results)

        results.put.assert_called_once_with(("startup_error", None, "Embedding model could not be loaded."))
        commands.get.assert_not_called()

    def test_worker_skips_unknown_commands_and_returns_ranked_results(self):
        commands = MagicMock()
        commands.get.side_effect = [
            ("unknown", 1, "ignored"),
            ("rank", 2, "query"),
            ("stop", None, None),
        ]
        results = MagicMock()
        hits = [{"description": "example"}]

        with (
            patch("pyopl.genai.rag_helper._load_model", return_value=object()),
            patch("pyopl.genai.rag_helper.rank_problem_descriptions", return_value=hits) as rank,
        ):
            exemplar_ranking_worker._run_ranking_worker(["models"], commands, results)

        self.assertEqual(
            results.put.call_args_list,
            [
                unittest.mock.call(("ready", None, None)),
                unittest.mock.call(("success", 2, hits)),
            ],
        )
        rank.assert_called_once_with(query="query", models_dir=["models"], top_k=0)

    def test_worker_reports_ranking_and_startup_exceptions(self):
        commands = MagicMock()
        commands.get.side_effect = [("rank", 3, "query"), ("stop", None, None)]
        results = MagicMock()

        with (
            patch("pyopl.genai.rag_helper._load_model", return_value=object()),
            patch("pyopl.genai.rag_helper.rank_problem_descriptions", side_effect=ValueError("bad query")),
        ):
            exemplar_ranking_worker._run_ranking_worker(["models"], commands, results)

        self.assertEqual(
            results.put.call_args_list,
            [
                unittest.mock.call(("ready", None, None)),
                unittest.mock.call(("error", 3, "ValueError: bad query")),
            ],
        )

        with (patch("pyopl.genai.rag_helper._load_model", side_effect=RuntimeError("startup failed")),):
            exemplar_ranking_worker._run_ranking_worker(["models"], commands, results)

        self.assertEqual(results.put.call_args.args[0], ("startup_error", None, "RuntimeError: startup failed"))

    def test_nonblocking_close_does_not_join_on_calling_thread(self):
        worker = ExemplarRankingWorker([Path("models")])
        process = MagicMock()
        process.is_alive.return_value = True
        commands = MagicMock()
        results = MagicMock()
        worker._process = process
        worker._commands = commands
        worker._results = results

        with patch("pyopl.genai.exemplar_ranking_worker.threading.Thread") as thread_class:
            worker.close(wait=False)

        process.terminate.assert_not_called()
        process.join.assert_not_called()
        thread_class.assert_called_once_with(
            target=worker._cleanup,
            args=(process, commands, results, True),
        )
        thread_class.return_value.start.assert_called_once_with()

    def test_async_cleanup_requests_graceful_worker_shutdown(self):
        process = MagicMock()
        process.is_alive.side_effect = [True, False]
        commands = MagicMock()
        results = MagicMock()

        ExemplarRankingWorker._cleanup(process, commands, results, True)

        commands.put.assert_called_once_with(("stop", None, None))
        process.join.assert_called_once_with(timeout=5.0)
        process.terminate.assert_not_called()
        commands.close.assert_called_once_with()
        commands.cancel_join_thread.assert_called_once_with()
        results.close.assert_called_once_with()
        results.cancel_join_thread.assert_called_once_with()

    def test_cleanup_terminates_worker_that_does_not_stop(self):
        process = MagicMock()
        process.is_alive.return_value = True
        commands = MagicMock()
        results = MagicMock()

        ExemplarRankingWorker._cleanup(process, commands, results, False)

        process.join.assert_any_call(timeout=1.0)
        self.assertEqual(process.join.call_count, 2)
        process.terminate.assert_called_once_with()
        commands.put.assert_not_called()
        commands.close.assert_called_once_with()
        results.close.assert_called_once_with()

    def test_cleanup_continues_after_process_and_queue_errors(self):
        process = MagicMock()
        process.is_alive.return_value = False
        process.join.side_effect = OSError("join failed")
        commands = MagicMock()
        commands.close.side_effect = OSError("close failed")
        results = MagicMock()
        results.cancel_join_thread.side_effect = OSError("cancel failed")

        ExemplarRankingWorker._cleanup(process, commands, results, False)

        process.join.assert_called_once_with(timeout=1.0)
        commands.close.assert_called_once_with()
        results.close.assert_called_once_with()
        results.cancel_join_thread.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
