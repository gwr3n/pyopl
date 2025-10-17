import argparse
import contextlib
import sys
import unittest
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run unittest test suite. Run all discovered tests or a single test by name and output results to a file."
    )
    # Run a single test (e.g., 'test.test_problems.TestPyOPLProblems.test_complex_workforce_planning')
    parser.add_argument(
        "-t",
        "--test",
        help="Dotted test name to run a single test (module.Class.test or module.Class or module)",
    )
    # Discovery options (used when --test is not provided)
    parser.add_argument("--start-dir", default="test", help="Directory to start discovery from (default: %(default)s)")
    parser.add_argument(
        "--pattern", default="test*.py", help="Pattern to match test files during discovery (default: %(default)s)"
    )
    parser.add_argument("--top-level-dir", default=None, help="Top level directory of project (optional)")
    # Output and verbosity
    parser.add_argument(
        "-o", "--output", default="unittest_results.txt", help="Write test output to this file instead of stdout"
    )
    parser.add_argument("-v", "--verbosity", type=int, default=2, help="Verbosity level for unittest (default: %(default)s)")
    return parser.parse_args(argv)


def build_suite(args: argparse.Namespace) -> unittest.TestSuite:
    if args.test:
        # Load a specific test by dotted name
        return unittest.defaultTestLoader.loadTestsFromName(args.test)
    # Discover the full test suite
    return unittest.defaultTestLoader.discover(
        start_dir=args.start_dir, pattern=args.pattern, top_level_dir=args.top_level_dir
    )


def run_tests(suite: unittest.TestSuite, verbosity: int, output: str | None) -> bool:
    result = None

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            runner = unittest.TextTestRunner(stream=f, verbosity=verbosity)
            result = runner.run(suite)
            f.flush()
    else:
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=verbosity)
        result = runner.run(suite)

    return bool(result and result.wasSuccessful())


def main(argv=None) -> int:
    args = parse_args(argv)
    suite = build_suite(args)
    success = run_tests(suite, verbosity=args.verbosity, output=args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
