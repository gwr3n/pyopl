import contextlib
import sys
import unittest

if __name__ == "__main__":
    # Run all tests discovered under the "test" package
    suite = unittest.defaultTestLoader.discover("test")

    # To run only a single test, use:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "test.test_problems.TestPyOPLProblems.test_complex_workforce_planning"
    )

    with open("unittest_results.txt", "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        result = runner.run(suite)
        f.flush()

    sys.exit(0 if result.wasSuccessful() else 1)
