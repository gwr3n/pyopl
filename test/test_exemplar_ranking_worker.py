import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pyopl.genai.exemplar_ranking_worker import ExemplarRankingWorker


class TestExemplarRankingWorker(unittest.TestCase):
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

        process.terminate.assert_called_once_with()
        process.join.assert_not_called()
        thread_class.assert_called_once_with(
            target=worker._cleanup,
            args=(process, commands, results, False),
            daemon=True,
        )
        thread_class.return_value.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
