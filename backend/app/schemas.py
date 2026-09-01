"""Request/response models for the VQE API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParamSpecOut(BaseModel):
    name: str
    label: str
    default: float
    min: float
    max: float


class CoefficientSpecOut(ParamSpecOut):
    pass


class ScenarioOut(BaseModel):
    id: str
    label: str
    hamiltonian: str
    n_qubits: int
    params: list[ParamSpecOut]
    coefficient: CoefficientSpecOut | None
    true_ground_energy: float


class SamplingRequest(BaseModel):
    """Fields shared by every endpoint that draws shots.

    readout_error is the symmetric measurement bit-flip probability, capped at
    the 0.25 top of the 5/10/15/20% grid the noise study sweeps; None keeps
    whichever value the scenario's own script declares. seed is the simulator
    seed; None means the server draws one and echoes it back.
    """

    readout_error: float | None = Field(default=None, ge=0.0, le=0.25)
    seed: int | None = Field(default=None, ge=0, lt=2**31)


class SamplingResponse(BaseModel):
    """The settings a result was actually produced with, so any figure taken
    from it can be regenerated exactly."""

    readout_error: float
    seed: int


class EvaluateRequest(SamplingRequest):
    params: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None


class EvaluateResponse(SamplingResponse):
    exact: float
    ideal_sampled: float
    noisy_sampled: float
    fake_sampled: float


class LandscapeRequest(SamplingRequest):
    sweep_param_index: int = Field(default=0, ge=0, le=7)
    fixed_params: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None
    # Every point costs four simulator runs. 24 keeps the heaviest scenario
    # (2q_xx_yy_zz, ~0.3s/point) under ~7s; the cap stops one request from
    # pinning the VPS CPU.
    n_points: int = Field(default=24, ge=4, le=120)


class LandscapeResponse(SamplingResponse):
    x: list[float]
    exact: list[float]
    ideal_sampled: list[float]
    noisy_sampled: list[float]
    fake_sampled: list[float]


class VqeRequest(BaseModel):
    x0: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None


class HistoryPoint(BaseModel):
    params: list[float]
    energy: float


class VqeResponse(BaseModel):
    optimal_params: list[float]
    optimal_energy: float
    true_ground_energy: float
    iterations: int
    history: list[HistoryPoint]


class HistogramRequest(SamplingRequest):
    params: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None


class SampledProbs(BaseModel):
    mean: list[float]
    std: list[float]


class HistogramResponse(SamplingResponse):
    bitstrings: list[str]
    exact: list[float]
    ideal: SampledProbs
    noisy: SampledProbs
    fake: SampledProbs
    n_shots: int
    n_reps: int
