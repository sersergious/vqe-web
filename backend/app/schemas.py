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


class EvaluateRequest(BaseModel):
    params: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None


class EvaluateResponse(BaseModel):
    exact: float
    ideal_sampled: float
    noisy_sampled: float
    fake_sampled: float


class LandscapeRequest(BaseModel):
    sweep_param_index: int = Field(default=0, ge=0, le=7)
    fixed_params: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None
    # Every point costs four simulator runs. 24 keeps the heaviest scenario
    # (2q_xx_yy_zz, ~0.3s/point) under ~7s; the cap stops one request from
    # pinning the VPS CPU.
    n_points: int = Field(default=24, ge=4, le=120)


class LandscapeResponse(BaseModel):
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


class HistogramRequest(BaseModel):
    params: list[float] = Field(min_length=1, max_length=8)
    coefficient: float | None = None


class SampledProbs(BaseModel):
    mean: list[float]
    std: list[float]


class HistogramResponse(BaseModel):
    bitstrings: list[str]
    exact: list[float]
    ideal: SampledProbs
    noisy: SampledProbs
    fake: SampledProbs
    n_shots: int
    n_reps: int
