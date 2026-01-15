#!/usr/bin/env python3
"""Python2-safe wrapper for training entrypoint."""

from __future__ import print_function

import os
import runpy
import sys


def _run_py3() -> None:
    script = os.path.join(os.path.dirname(__file__), "train_main.py")
    runpy.run_path(script, run_name="__main__")


def main() -> None:
    if sys.version_info < (3, 8):
        python3 = os.environ.get("PYTHON3", "python3")
        script = os.path.join(os.path.dirname(__file__), "train_main.py")
        os.execvp(python3, [python3, script] + sys.argv[1:])
    _run_py3()


if __name__ == "__main__":
    main()
