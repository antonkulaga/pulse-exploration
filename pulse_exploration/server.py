"""Laptop server for the prototype.

- Watch probe app: one `PUT /runs/{run}/{file}` per file of a run, stored under data/runs/.
- Phone camera PPG: `POST /camera/video` and `POST /camera/trace` (see camera_api), stored under
  data/camera/.
- Phone pages: `/`, `/upload`, `/capture` (see pages_api; /capture needs HTTPS, see
  scripts/make_cert.py).

Run with:

    uv run uvicorn pulse_exploration.server:app --host 0.0.0.0 --port 8765

There is no authentication: run it only on a trusted LAN. Recordings are personal health data;
data/ is gitignored.
"""

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from pulse_exploration import camera_api
from pulse_exploration.pages_api import router as pages_router
from pulse_exploration.watch_api import router as watch_router

DATA = Path(__file__).resolve().parent.parent / "data" / "runs"
SAFE = re.compile(r"^[A-Za-z0-9._-]+$")

app = FastAPI(title="pulse-exploration")
app.include_router(camera_api.router)
app.include_router(pages_router)
app.include_router(watch_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/runs")
def runs() -> dict:
    if not DATA.exists():
        return {"runs": []}
    return {"runs": sorted(p.name for p in DATA.iterdir() if p.is_dir())}


@app.put("/runs/{run}/{name}")
async def put_file(run: str, name: str, request: Request) -> dict:
    if not SAFE.match(run) or not SAFE.match(name) or name.startswith("."):
        raise HTTPException(400, "bad run or file name")
    target = DATA / run
    target.mkdir(parents=True, exist_ok=True)
    body = await request.body()
    (target / name).write_bytes(body)
    return {"run": run, "file": name, "bytes": len(body)}
