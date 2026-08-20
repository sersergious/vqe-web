"""FastAPI service exposing the vqe/ experiment scripts as a JSON API.

Also serves the built frontend, so the whole app is one container on one origin:
no CORS, no separate proxy hop, and one set of credentials guarding both.

Runs as a long-lived process (not serverless): simulators stay cached between
requests and handlers may take tens of seconds without hitting a timeout.
"""

from __future__ import annotations

import base64
import binascii
import logging
import math
import os
import secrets
from pathlib import Path
from typing import Sequence

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles

from . import scenarios as sc
from .schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HistogramRequest,
    HistogramResponse,
    LandscapeRequest,
    LandscapeResponse,
    ScenarioOut,
    VqeRequest,
    VqeResponse,
)

log = logging.getLogger("uvicorn.error")

AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "")
AUTH_ENABLED = bool(AUTH_USERNAME and AUTH_PASSWORD)

# Written here by the Docker build; absent when running the API alone in dev.
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "static"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not AUTH_ENABLED:
        log.warning(
            "AUTH_USERNAME/AUTH_PASSWORD are not set — this instance is OPEN to "
            "anyone who can reach it. Set both before exposing it publicly."
        )
    yield


app = FastAPI(title="VQE Simulation API", version="2.0.0", lifespan=lifespan)


def _authorized(header: str | None) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    # Compare both halves unconditionally so timing does not reveal which failed.
    valid_user = secrets.compare_digest(username.encode(), AUTH_USERNAME.encode())
    valid_password = secrets.compare_digest(password.encode(), AUTH_PASSWORD.encode())
    return valid_user and valid_password


@app.middleware("http")
async def enforce_basic_auth(request: Request, call_next):
    """Guards the API and the frontend alike — everything except the health check."""
    if (
        AUTH_ENABLED
        and request.url.path != "/api/health"
        and not _authorized(request.headers.get("authorization"))
    ):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="VQE Explorer"'},
        )
    return await call_next(request)


def get_scenario(scenario_id: str) -> sc.Scenario:
    scenario = sc.SCENARIOS.get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario '{scenario_id}'")
    return scenario


def validate(
    scenario: sc.Scenario, params: Sequence[float], coefficient: float | None
) -> float | None:
    """Check the request against the scenario before handing values to Qiskit."""
    if len(params) != len(scenario.params):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Scenario '{scenario.id}' takes {len(scenario.params)} parameters, "
                f"got {len(params)}"
            ),
        )
    if any(not math.isfinite(value) for value in params):
        raise HTTPException(status_code=422, detail="Parameters must be finite numbers")

    spec = scenario.coefficient
    if spec is None:
        return None
    value = spec.default if coefficient is None else float(coefficient)
    if not math.isfinite(value) or not (spec.min <= value <= spec.max):
        raise HTTPException(
            status_code=422,
            detail=f"'{spec.name}' must be between {spec.min} and {spec.max}",
        )
    return value


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenarios", response_model=list[ScenarioOut])
def list_scenarios() -> list[dict]:
    return [
        {
            "id": s.id,
            "label": s.label,
            "hamiltonian": s.hamiltonian,
            "n_qubits": s.n_qubits,
            "params": [vars(p) for p in s.params],
            "coefficient": vars(s.coefficient) if s.coefficient else None,
            "true_ground_energy": s.ground_energy(
                s.coefficient.default if s.coefficient else None
            ),
        }
        for s in sc.SCENARIOS.values()
    ]


@app.post("/api/scenarios/{scenario_id}/evaluate", response_model=EvaluateResponse)
def evaluate(scenario_id: str, body: EvaluateRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.params, body.coefficient)
    return sc.all_energies(scenario, body.params, coefficient)


@app.post("/api/scenarios/{scenario_id}/landscape", response_model=LandscapeResponse)
def landscape(scenario_id: str, body: LandscapeRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.fixed_params, body.coefficient)
    if body.sweep_param_index >= len(scenario.params):
        raise HTTPException(
            status_code=422,
            detail=f"sweep_param_index out of range for scenario '{scenario.id}'",
        )
    return sc.landscape(
        scenario,
        body.sweep_param_index,
        body.fixed_params,
        coefficient,
        body.n_points,
    )


@app.post("/api/scenarios/{scenario_id}/vqe", response_model=VqeResponse)
def vqe(scenario_id: str, body: VqeRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.x0, body.coefficient)
    return sc.run_vqe(scenario, body.x0, coefficient)


@app.post("/api/scenarios/{scenario_id}/histogram", response_model=HistogramResponse)
def histogram(scenario_id: str, body: HistogramRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.params, body.coefficient)
    return sc.histogram(scenario, body.params, coefficient)


# Mounted last: this catches every path the API routes above did not claim.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
