"""HTTP endpoint for watch run analysis (include with ``app.include_router(watch_api.router)``).

* ``GET /runs/{run}/analysis`` - ``watch_session.analyze_run`` of ``data/runs/<run>/``. Computed on
  the first request and cached as ``analysis.json`` in the run directory; ``?refresh=1``
  recomputes. The cache is also recomputed when a file of the run is newer than it (the watch
  uploads a run file by file) or when it was written by another analysis version.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from pulse_exploration import watch_session

router = APIRouter(tags=["watch"])

DATA = Path(__file__).resolve().parent.parent / "data" / "runs"
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")  # same rule as server.py
INPUTS = {".csv", ".tsv", ".json", ".txt"}  # files a run's analysis is computed from


def _run_dir(run: str) -> Path:
    if not SAFE.match(run) or run.startswith("."):
        raise HTTPException(400, "bad run name")
    path = DATA / run
    if not path.is_dir():
        raise HTTPException(404, f"no run {run}")
    return path


def _cached(run: Path) -> dict | None:
    path = run / watch_session.ANALYSIS_FILE
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if not isinstance(cached, dict) or cached.get("analysis_version") != watch_session.ANALYSIS_VERSION:
        return None
    written = path.stat().st_mtime
    inputs = (p for p in run.iterdir() if p.suffix in INPUTS and p.name != path.name and p.is_file())
    if any(p.stat().st_mtime > written for p in inputs):
        return None
    return cached


@router.get("/runs/{run}/analysis")
def run_analysis(run: str, refresh: bool = False) -> dict:
    """Analysis of one watch run (see watch_session for the JSON shape)."""
    path = _run_dir(run)
    if not refresh and (cached := _cached(path)) is not None:
        return cached
    return watch_session.analyze_run(path)
