"""
Custom build step: compiles the hdcd C shared library via the top-level
Makefile and copies it into the hdcd package directory before the
Python sources are collected, so `pip install .` produces a
self-contained installation without requiring the user to run `make`
by hand first (spec section 31 Milestone 10: "installable package").
"""

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def _shared_lib_name():
    return "libhdcd.dylib" if platform.system() == "Darwin" else "libhdcd.so"


class BuildPy(_build_py):
    def run(self):
        lib_name = _shared_lib_name()
        subprocess.run(["make", "-C", str(REPO_ROOT), "shared"], check=True)
        built_lib = REPO_ROOT / "build" / lib_name
        if not built_lib.exists():
            raise RuntimeError(f"expected shared library not found after build: {built_lib}")
        dest_dir = PACKAGE_DIR / "hdcd"
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(built_lib, dest_dir / lib_name)
        super().run()


setup(cmdclass={"build_py": BuildPy})
