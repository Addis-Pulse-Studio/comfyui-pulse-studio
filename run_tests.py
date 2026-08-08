#!/usr/bin/env python3
"""Run the headless test suite.

Deliberately stdlib-only (unittest, not pytest): the core of this project must
be testable inside a bare ComfyUI environment with no extra install, and on a
machine with no GPU. Nothing under omni_director/ imports torch or comfy.

    python3 run_tests.py [-v]
"""

import sys
import unittest


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", top_level_dir=".")
    verbosity = 2 if "-v" in sys.argv else 1
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
