"""Data and result locations for the release.

Every location is configurable through an environment variable:

  DC_CACHE_ROOT      generation records (default: <release>/data)
  DC_RESULTS_ROOT    outputs written by the scripts (default: <release>/results)
  DC_REFERENCE_ROOT  reference tables read by the tests (default: <release>/reference)
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent


def _root(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, default)).expanduser().resolve()


CACHE_ROOT = _root("DC_CACHE_ROOT", PROJECT_ROOT / "data")
RESULTS_ROOT = _root("DC_RESULTS_ROOT", PROJECT_ROOT / "results")
REFERENCE_ROOT = _root("DC_REFERENCE_ROOT", PROJECT_ROOT / "reference")

_LOCATIONS = {
    # K=32 self-consistency records of the five primary cells
    "sc_records": CACHE_ROOT / "sc_records",
    # K=32 self-consistency records of the reasoning model
    "reasoning_records": CACHE_ROOT / "reasoning_records",
    # K=8 self-consistency and prompt-template records of the 5 x 12 matrix
    "cache": CACHE_ROOT / "capability_matrix",
    "results": RESULTS_ROOT,
    "tables": RESULTS_ROOT / "tables",
    "cross_benchmark": RESULTS_ROOT / "cross_benchmark",
    "figures": RESULTS_ROOT / "figures",
    "reference": REFERENCE_ROOT,
}


def experiment_dir(name: str) -> Path:
    """Resolve a logical data or result location."""
    try:
        return _LOCATIONS[name]
    except KeyError:
        known = ", ".join(sorted(_LOCATIONS))
        raise KeyError(f"unknown location {name!r}; known locations: {known}") from None


def record_path(path: Path) -> str:
    """Render a path relative to the release root when it is inside it."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


CACHE_DIR = experiment_dir("cache")
RESULTS_DIR = experiment_dir("results")
