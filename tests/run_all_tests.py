"""
run_all_tests.py
================
Test runner for the miniGPT test suite.

Usage
-----
    python tests/run_all_tests.py              # run all tests
    python tests/run_all_tests.py -v           # verbose
    python tests/run_all_tests.py --quick      # tokenizer only (~1s)
    python tests/run_all_tests.py --module data     # one module
    python tests/run_all_tests.py --module neural_network
"""

import argparse
import os
import sys
import time
import unittest

# Make sure the project root is on the path regardless of where this is called from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODULES = [
    "tokenizer",
    "data",
    "neural_network",
    "model",
    "build_dataset",
    "integration",
    "performance",
]


def _discover(pattern: str, verbose: bool) -> unittest.TestResult:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    loader    = unittest.TestLoader()
    suite     = loader.discover(tests_dir, pattern=pattern)
    runner    = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    return runner.run(suite)


def _summary(result: unittest.TestResult, elapsed: float) -> int:
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print()
    print("=" * 60)
    print(f"  Ran {result.testsRun} tests in {elapsed:.2f}s")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {len(result.failures)}")
    print(f"  Errors:  {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 60)
    if result.wasSuccessful():
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(
        description="miniGPT test runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each test name")
    parser.add_argument("-q", "--quick", action="store_true",
                        help="Run only tokenizer tests (~1s)")
    parser.add_argument("--module", choices=MODULES,
                        help="Run only the named module")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if args.quick:
        result = _discover("test_tokenizer.py", args.verbose)
    elif args.module:
        result = _discover(f"test_{args.module}.py", args.verbose)
    else:
        result = _discover("test_*.py", args.verbose)

    sys.exit(_summary(result, time.perf_counter() - t0))


if __name__ == "__main__":
    main()