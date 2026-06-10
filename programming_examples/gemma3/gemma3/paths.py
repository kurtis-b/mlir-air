"""Canonical filesystem paths for the Gemma3 programming example."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = PACKAGE_DIR.parent
PROGRAMMING_EXAMPLES_DIR = EXAMPLE_ROOT.parent
REPO_ROOT = PROGRAMMING_EXAMPLES_DIR.parent

DATA_DIR = EXAMPLE_ROOT / "data"
DOCS_DIR = EXAMPLE_ROOT / "docs"
RESULTS_DIR = EXAMPLE_ROOT / "results"
AIE_KERNELS_DIR = EXAMPLE_ROOT / "aie_kernels"
TESTS_DIR = EXAMPLE_ROOT / "tests"

PAPER_TARGETS_JSON = DATA_DIR / "paper_targets.json"


def aie_kernel_source(filename: str) -> Path:
    return AIE_KERNELS_DIR / filename


def result_path(filename: str) -> Path:
    return RESULTS_DIR / filename
