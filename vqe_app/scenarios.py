"""Adapts the nine standalone VQE scripts in vqe/ to one uniform interface.

The scripts are imported unmodified. They fall into four calling conventions,
so each Scenario declares a `family` and the call_* helpers below branch on it:

  1q_scalar     vqe_1q_{x,y,z}_noise                 one angle, H fixed
  1q_pair       vqe_1q_x_plus_z_noise                (theta, phi), H fixed
  1q_pair_coeff vqe_1q_{b_dot_z,cosx_plus_sinz}      (theta, phi), H tunable
  2q            vqe_2q_{zz,zi_plus_iz,xx_yy_zz}      8 params, H module constant
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from math import pi
from pathlib import Path
from types import ModuleType
from typing import Callable, Sequence

# The scripts import matplotlib.pyplot at module scope; pin a headless backend
# before that happens so importing them works inside a container.
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the repo root importable so `vqe.*` resolves when uvicorn runs from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vqe import (  # noqa: E402
    vqe_1q_b_dot_z_noise,
    vqe_1q_cosx_plus_sinz_noise,
    vqe_1q_x_noise,
    vqe_1q_x_plus_z_noise,
    vqe_1q_y_noise,
    vqe_1q_z_noise,
    vqe_2q_xx_yy_zz_noise,
    vqe_2q_zi_plus_iz_noise,
    vqe_2q_zz_noise,
)

FAMILY_1Q_SCALAR = "1q_scalar"
FAMILY_1Q_PAIR = "1q_pair"
FAMILY_1Q_PAIR_COEFF = "1q_pair_coeff"
FAMILY_2Q = "2q"

BITSTRINGS_1Q = ("0", "1")
BITSTRINGS_2Q = ("00", "01", "10", "11")

TWO_PI = 2.0 * pi


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    default: float
    min: float = 0.0
    max: float = TWO_PI


@dataclass(frozen=True)
class CoefficientSpec:
    name: str
    label: str
    default: float
    min: float
    max: float


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    hamiltonian: str
    family: str
    n_qubits: int
    module: ModuleType
    params: tuple[ParamSpec, ...]
    coefficient: CoefficientSpec | None
    # Ground energy depends on the coefficient for B*Z; constant everywhere else.
    ground_energy: Callable[[float | None], float]
    # Which fake device this script's build_fake_simulator() snapshots. Scenarios
    # naming the same one share a single simulator — see _fake_simulator.
    fake_backend: str


def _angles(*specs: tuple[str, str, float]) -> tuple[ParamSpec, ...]:
    return tuple(ParamSpec(name, label, default) for name, label, default in specs)


# The 2q scripts seed COBYLA from this exact draw; keep the same starting point.
_X0_2Q = np.random.default_rng(42).uniform(0.1, 0.5, size=8)
_PARAMS_2Q = tuple(
    ParamSpec(name, name, float(_X0_2Q[i]))
    for i, name in enumerate(["t0", "t1", "t2", "t3", "p0", "p1", "p2", "p3"])
)


def _const(value: float) -> Callable[[float | None], float]:
    return lambda _coefficient: value


_SCENARIO_LIST: tuple[Scenario, ...] = (
    Scenario(
        id="1q_z",
        label="Single qubit — Z",
        hamiltonian="H = Z",
        family=FAMILY_1Q_SCALAR,
        n_qubits=1,
        module=vqe_1q_z_noise,
        params=_angles(("theta", "θ", 1.0)),
        coefficient=None,
        ground_energy=_const(-1.0),
        fake_backend="FakeManilaV2",
    ),
    Scenario(
        id="1q_x",
        label="Single qubit — X",
        hamiltonian="H = X",
        family=FAMILY_1Q_SCALAR,
        n_qubits=1,
        module=vqe_1q_x_noise,
        params=_angles(("theta", "θ", 1.0)),
        coefficient=None,
        ground_energy=_const(-1.0),
        fake_backend="FakeManilaV2",
    ),
    Scenario(
        id="1q_y",
        label="Single qubit — Y",
        hamiltonian="H = Y",
        family=FAMILY_1Q_SCALAR,
        n_qubits=1,
        module=vqe_1q_y_noise,
        params=_angles(("theta", "θ", 1.0)),
        coefficient=None,
        ground_energy=_const(-1.0),
        fake_backend="FakeManilaV2",
    ),
    Scenario(
        id="1q_x_plus_z",
        label="Single qubit — X + Z",
        hamiltonian="H = X + Z",
        family=FAMILY_1Q_PAIR,
        n_qubits=1,
        module=vqe_1q_x_plus_z_noise,
        params=_angles(("theta", "θ", 1.0), ("phi", "φ", 0.5)),
        coefficient=None,
        ground_energy=_const(-(2.0**0.5)),
        fake_backend="FakeManilaV2",
    ),
    Scenario(
        id="1q_b_dot_z",
        label="Single qubit — spin in a field (−B·Z)",
        hamiltonian="H = −B·Z",
        family=FAMILY_1Q_PAIR_COEFF,
        n_qubits=1,
        module=vqe_1q_b_dot_z_noise,
        params=_angles(("theta", "θ", 1.0), ("phi", "φ", 0.5)),
        coefficient=CoefficientSpec(
            name="B",
            label="B (field strength)",
            default=float(vqe_1q_b_dot_z_noise.B),
            min=-3.0,
            max=3.0,
        ),
        ground_energy=lambda coefficient: -abs(float(coefficient)),
        fake_backend="FakeManilaV2",
    ),
    Scenario(
        id="1q_cosx_plus_sinz",
        label="Single qubit — cos(α)·X + sin(α)·Z",
        hamiltonian="H = cos(α)·X + sin(α)·Z",
        family=FAMILY_1Q_PAIR_COEFF,
        n_qubits=1,
        module=vqe_1q_cosx_plus_sinz_noise,
        params=_angles(("theta", "θ", 1.0), ("phi", "φ", 0.5)),
        coefficient=CoefficientSpec(
            name="alpha",
            label="α (mixing angle)",
            default=float(vqe_1q_cosx_plus_sinz_noise.ALPHA),
            min=0.0,
            max=TWO_PI,
        ),
        ground_energy=_const(-1.0),
        fake_backend="FakeManilaV2",
    ),
    Scenario(
        id="2q_zz",
        label="Two qubits — Z⊗Z",
        hamiltonian="H = Z⊗Z",
        family=FAMILY_2Q,
        n_qubits=2,
        module=vqe_2q_zz_noise,
        params=_PARAMS_2Q,
        coefficient=None,
        ground_energy=_const(float(vqe_2q_zz_noise.TRUE_GROUND_ENERGY)),
        fake_backend="FakeSherbrooke",
    ),
    Scenario(
        id="2q_zi_plus_iz",
        label="Two qubits — Z⊗I + I⊗Z",
        hamiltonian="H = Z⊗I + I⊗Z",
        family=FAMILY_2Q,
        n_qubits=2,
        module=vqe_2q_zi_plus_iz_noise,
        params=_PARAMS_2Q,
        coefficient=None,
        ground_energy=_const(float(vqe_2q_zi_plus_iz_noise.TRUE_GROUND_ENERGY)),
        fake_backend="FakeSherbrooke",
    ),
    Scenario(
        id="2q_xx_yy_zz",
        label="Two qubits — Heisenberg (X⊗X + Y⊗Y + Z⊗Z)",
        hamiltonian="H = X⊗X + Y⊗Y + Z⊗Z",
        family=FAMILY_2Q,
        n_qubits=2,
        module=vqe_2q_xx_yy_zz_noise,
        params=_PARAMS_2Q,
        coefficient=None,
        ground_energy=_const(float(vqe_2q_xx_yy_zz_noise.TRUE_GROUND_ENERGY)),
        fake_backend="FakeSherbrooke",
    ),
)

SCENARIOS: dict[str, Scenario] = {s.id: s for s in _SCENARIO_LIST}


@dataclass(frozen=True)
class Simulators:
    ideal: object
    noisy: object
    fake: object


_FAKE_SIMULATORS: dict[str, object] = {}


def _fake_simulator(scenario: Scenario) -> object:
    """One simulator per fake device, shared by every scenario that names it.

    All six 1-qubit scripts snapshot FakeManilaV2 and all three 2-qubit ones
    snapshot FakeSherbrooke, identically. Building one per scenario instead costs
    ~58 MB each for Sherbrooke, and Aer does not hand that memory back when the
    simulator is released — so an eviction policy cannot recover it and the only
    fix is not allocating it. Sharing is safe: these are stateless across .run().
    """
    cached = _FAKE_SIMULATORS.get(scenario.fake_backend)
    if cached is None:
        cached = scenario.module.build_fake_simulator()
        _FAKE_SIMULATORS[scenario.fake_backend] = cached
    return cached


@lru_cache(maxsize=None)
def simulators(scenario_id: str) -> Simulators:
    """Built once per scenario — AerSimulator.from_backend(FakeSherbrooke) is slow."""
    scenario = SCENARIOS[scenario_id]
    return Simulators(
        ideal=scenario.module.build_ideal_simulator(),
        noisy=scenario.module.build_noisy_simulator(),
        fake=_fake_simulator(scenario),
    )


@lru_cache(maxsize=None)
def _ansatz(scenario_id: str):
    """(circuit, param_list) — param_list is None for the 1q families."""
    scenario = SCENARIOS[scenario_id]
    if scenario.family == FAMILY_2Q:
        return scenario.module.build_ansatz()
    return scenario.module.build_ansatz(), None


def _prepare(scenario: Scenario, coefficient: float | None):
    """Returns (ansatz, ctx) where ctx is the third arg the script's functions want:
    the parameter list for 2q scripts, the Hamiltonian for 1q ones."""
    ansatz, param_list = _ansatz(scenario.id)
    if scenario.family == FAMILY_2Q:
        return ansatz, param_list
    if scenario.coefficient is not None:
        return ansatz, scenario.module.build_hamiltonian(coefficient)
    return ansatz, scenario.module.build_hamiltonian()


def _pack(scenario: Scenario, params: Sequence[float]):
    """The 1q_scalar scripts take a bare float; every other family takes a vector."""
    if scenario.family == FAMILY_1Q_SCALAR:
        return float(params[0])
    return np.asarray(params, dtype=float)


def exact_energy(
    scenario: Scenario, params: Sequence[float], coefficient: float | None = None
) -> float:
    ansatz, ctx = _prepare(scenario, coefficient)
    return float(scenario.module.exact_expectation(_pack(scenario, params), ansatz, ctx))


def sampled_energy(
    scenario: Scenario,
    params: Sequence[float],
    simulator,
    coefficient: float | None = None,
) -> float:
    packed = _pack(scenario, params)
    if scenario.family == FAMILY_1Q_PAIR_COEFF:
        # These two take the Hamiltonian coefficient as a third positional arg.
        return float(scenario.module.sampled_expectation(packed, simulator, coefficient))
    return float(scenario.module.sampled_expectation(packed, simulator))


def all_energies(
    scenario: Scenario, params: Sequence[float], coefficient: float | None = None
) -> dict[str, float]:
    sims = simulators(scenario.id)
    return {
        "exact": exact_energy(scenario, params, coefficient),
        "ideal_sampled": sampled_energy(scenario, params, sims.ideal, coefficient),
        "noisy_sampled": sampled_energy(scenario, params, sims.noisy, coefficient),
        "fake_sampled": sampled_energy(scenario, params, sims.fake, coefficient),
    }


def landscape(
    scenario: Scenario,
    sweep_index: int,
    fixed_params: Sequence[float],
    coefficient: float | None = None,
    n_points: int = 40,
) -> dict[str, list[float]]:
    """Sweep one parameter over [0, 2π] with the others held at fixed_params."""
    xs = np.linspace(0.0, TWO_PI, n_points)
    series: dict[str, list[float]] = {
        "x": xs.tolist(),
        "exact": [],
        "ideal_sampled": [],
        "noisy_sampled": [],
        "fake_sampled": [],
    }
    working = list(fixed_params)
    for x in xs:
        working[sweep_index] = float(x)
        for key, value in all_energies(scenario, working, coefficient).items():
            series[key].append(value)
    return series


def run_vqe(
    scenario: Scenario, x0: Sequence[float], coefficient: float | None = None
) -> dict:
    ansatz, ctx = _prepare(scenario, coefficient)
    module = scenario.module

    if scenario.family == FAMILY_2Q:
        optimal, energy, history = module.run_vqe(
            ansatz, ctx, np.asarray(x0, dtype=float)
        )
    elif scenario.family == FAMILY_1Q_SCALAR:
        optimal, energy, history = module.run_vqe(ansatz, ctx, theta_init=float(x0[0]))
    else:
        optimal, energy, history = module.run_vqe(
            ansatz, ctx, theta_init=float(x0[0]), phi_init=float(x0[1])
        )

    return {
        "optimal_params": np.atleast_1d(optimal).astype(float).tolist(),
        "optimal_energy": float(energy),
        "true_ground_energy": scenario.ground_energy(coefficient),
        "iterations": len(history),
        "history": [
            {
                "params": np.atleast_1d(p).astype(float).tolist(),
                "energy": float(value),
            }
            for p, value in history
        ],
    }


def histogram(
    scenario: Scenario, params: Sequence[float], coefficient: float | None = None
) -> dict:
    """Measurement-outcome probabilities: exact plus mean/std over repeated shot runs.

    Shot counts and repetitions come from each script's own hardcoded values and
    are echoed back so the frontend can label the chart.
    """
    ansatz, ctx = _prepare(scenario, coefficient)
    sims = simulators(scenario.id)
    data = scenario.module.build_histogram_data(
        _pack(scenario, params), sims.ideal, sims.noisy, sims.fake, ansatz, ctx
    )

    if scenario.family == FAMILY_2Q:
        keys = list(BITSTRINGS_2Q)
        pick = lambda d: [float(d[k]) for k in keys]  # noqa: E731
        return {
            "bitstrings": keys,
            "exact": pick(data.p_exact),
            "ideal": {"mean": pick(data.p_ideal_mean), "std": pick(data.p_ideal_std)},
            "noisy": {"mean": pick(data.p_noisy_mean), "std": pick(data.p_noisy_std)},
            "fake": {"mean": pick(data.p_fake_mean), "std": pick(data.p_fake_std)},
            "n_shots": int(data.n_shots),
            "n_reps": int(data.n_reps),
        }

    return {
        "bitstrings": list(BITSTRINGS_1Q),
        "exact": [float(data.p0_exact), float(data.p1_exact)],
        "ideal": {
            "mean": [float(data.p0_ideal_mean), float(data.p1_ideal_mean)],
            "std": [float(data.p0_ideal_std), float(data.p1_ideal_std)],
        },
        "noisy": {
            "mean": [float(data.p0_noisy_mean), float(data.p1_noisy_mean)],
            "std": [float(data.p0_noisy_std), float(data.p1_noisy_std)],
        },
        "fake": {
            "mean": [float(data.p0_fake_mean), float(data.p1_fake_mean)],
            "std": [float(data.p0_fake_std), float(data.p1_fake_std)],
        },
        "n_shots": int(data.n_shots),
        "n_reps": int(data.n_repetitions),
    }
