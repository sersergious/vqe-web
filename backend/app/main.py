"""FastAPI service exposing the vqe/ experiment scripts as a JSON API.

Will also serve the frontend if it has been built into backend/static, putting
the API and the page on one origin; otherwise the Next.js dev server proxies
/api/* here instead.

Simulators are cached in-process between requests, and a handler may take tens
of seconds — a landscape sweep runs four simulators at every point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import scenarios as sc
from .schemas import (
    EvaluateRequest,
    EvaluateResponse,
    HistogramRequest,
    HistogramResponse,
    LandscapeRequest,
    LandscapeResponse,
    SamplingRequest,
    ScenarioOut,
    VqeRequest,
    VqeResponse,
)

# Present only after `npm run build` is copied here; absent in the usual dev setup.
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(title="VQE Simulation API", version="2.0.0")


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


def sampling(scenario: sc.Scenario, body: SamplingRequest) -> dict[str, float | int]:
    """Resolve the shot-noise settings a response has to echo back.

    An omitted readout error falls back to the scenario script's own constant,
    and an omitted seed is drawn here — so every response names the exact
    settings that produced it and the run can be repeated.
    """
    return {
        "readout_error": (
            sc.readout_default(scenario)
            if body.readout_error is None
            else float(body.readout_error)
        ),
        "seed": sc.resolve_seed(body.seed),
    }


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
    options = sampling(scenario, body)
    return {
        **sc.all_energies(
            scenario,
            body.params,
            coefficient,
            options["readout_error"],
            options["seed"],
        ),
        **options,
    }


@app.post("/api/scenarios/{scenario_id}/landscape", response_model=LandscapeResponse)
def landscape(scenario_id: str, body: LandscapeRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.fixed_params, body.coefficient)
    if body.sweep_param_index >= len(scenario.params):
        raise HTTPException(
            status_code=422,
            detail=f"sweep_param_index out of range for scenario '{scenario.id}'",
        )
    options = sampling(scenario, body)
    return {
        **sc.landscape(
            scenario,
            body.sweep_param_index,
            body.fixed_params,
            coefficient,
            body.n_points,
            options["readout_error"],
            options["seed"],
        ),
        **options,
    }


@app.post("/api/scenarios/{scenario_id}/vqe", response_model=VqeResponse)
def vqe(scenario_id: str, body: VqeRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.x0, body.coefficient)
    return sc.run_vqe(scenario, body.x0, coefficient)


@app.post("/api/scenarios/{scenario_id}/histogram", response_model=HistogramResponse)
def histogram(scenario_id: str, body: HistogramRequest) -> dict:
    scenario = get_scenario(scenario_id)
    coefficient = validate(scenario, body.params, body.coefficient)
    options = sampling(scenario, body)
    return {
        **sc.histogram(
            scenario,
            body.params,
            coefficient,
            options["readout_error"],
            options["seed"],
        ),
        **options,
    }


# Mounted last: this catches every path the API routes above did not claim.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
