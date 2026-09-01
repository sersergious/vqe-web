#!/usr/bin/env python3

# VQE demonstration: isotropic Heisenberg model on two spins.
# Hamiltonian: H = X⊗X + Y⊗Y + Z⊗Z
# Qiskit SparsePauliOp strings (rightmost char = qubit 0):
#   "XX", "YY", "ZZ" with coefficient +1.0 each.
#
# Matrix in computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
#
#        |00⟩  |01⟩  |10⟩  |11⟩
#  |00⟩ [  1    0    0    0  ]
#  |01⟩ [  0   -1    2    0  ]
#  |10⟩ [  0    2   -1    0  ]
#  |11⟩ [  0    0    0    1  ]
#
# Spectrum:
#   E = +1 (triplet, 3-fold degenerate):
#       |00⟩,  |11⟩,  (|01⟩ + |10⟩)/√2
#   E = -3 (singlet, unique ground state):
#       |ψ⁻⟩ = (|01⟩ - |10⟩)/√2
#
# The ground state is maximally entangled (Bell state |ψ⁻⟩).
# Entanglement is necessary — no product state can reach E = -3.
#
# Sampled expectation — three measurement circuits:
#   <Z⊗Z>: measure in computational basis
#           parity = (N_00 - N_01 - N_10 + N_11) / N
#   <X⊗X>: apply H⊗H before measurement, then same parity formula
#   <Y⊗Y>: apply (Sdg·H)⊗(Sdg·H) before measurement, then same parity
#   <H>   = <XX> + <YY> + <ZZ>
#
# Optimization via COBYLA (scipy).
# Backends: StatevectorEstimator (exact), AerSimulator ideal,
#           manual noise model, FakeSherbrooke (127-qubit Eagle r3).

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp


# =============================================================================
# Hamiltonian and constants
# =============================================================================

N_QUBITS = 2
TRUE_GROUND_ENERGY = -3.0

# Qiskit string ordering: rightmost character = qubit 0.
HAMILTONIAN = SparsePauliOp.from_list(
    [
        ("XX", 1.0),
        ("YY", 1.0),
        ("ZZ", 1.0),
    ]
)


# =============================================================================
# Ansatz
# =============================================================================


def build_ansatz():
    """
    8-parameter Hardware Efficient Ansatz.

    q0: --Ry(t0)--Rz(p0)--*--Ry(t2)--Rz(p2)--
                           |
    q1: --Ry(t1)--Rz(p1)--X--Ry(t3)--Rz(p3)--

    The ground state |ψ⁻⟩ = (|01⟩ - |10⟩)/√2 is a maximally entangled
    Bell state. The CNOT is essential: no product state can reach E = -3.

    Parameter order: t0 t1 t2 t3 p0 p1 p2 p3.
    """
    t = [Parameter(f"t{i}") for i in range(4)]
    p = [Parameter(f"p{i}") for i in range(4)]

    qc = QuantumCircuit(N_QUBITS)

    # Layer 1
    qc.ry(t[0], 0)
    qc.rz(p[0], 0)
    qc.ry(t[1], 1)
    qc.rz(p[1], 1)

    # Entangler
    qc.cx(0, 1)

    # Layer 2
    qc.ry(t[2], 0)
    qc.rz(p[2], 0)
    qc.ry(t[3], 1)
    qc.rz(p[3], 1)

    return qc, t + p


# =============================================================================
# Simulator setup
# =============================================================================


def build_ideal_simulator():
    """Noiseless AerSimulator."""
    return AerSimulator()


def build_noisy_simulator(readout_p=0.02):
    """
    AerSimulator with manually constructed noise model.
    Single-qubit gates: depolarizing (p=0.005) + thermal relaxation.
    Two-qubit gate (cx): depolarizing (p=0.02) + thermal relaxation.
    Readout: 2% symmetric error per qubit.
    """
    noise_model = NoiseModel()
    t1, t2 = 100e-6, 80e-6
    gate_time_1q = 50e-9
    gate_time_2q = 300e-9

    dep_1q = depolarizing_error(0.005, 1)
    therm_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    err_1q = dep_1q.compose(therm_1q)
    noise_model.add_all_qubit_quantum_error(err_1q, ["ry", "rz", "h", "sdg"])

    dep_2q = depolarizing_error(0.02, 2)
    therm_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q)
    )
    err_2q = dep_2q.compose(therm_2q)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

    readout_error = ReadoutError(
        [[1 - readout_p, readout_p], [readout_p, 1 - readout_p]]
    )
    noise_model.add_all_qubit_readout_error(readout_error)

    return AerSimulator(noise_model=noise_model)


def build_fake_simulator():
    """AerSimulator from FakeSherbrooke (IBM Eagle r3, 127 qubits)."""
    return AerSimulator.from_backend(FakeSherbrooke())


# =============================================================================
# Expectation value
# =============================================================================


def exact_expectation(param_vals, ansatz, param_list):
    """
    Deterministic <XX + YY + ZZ> via StatevectorEstimator.
    param_vals: length-8 array ordered t0 t1 t2 t3 p0 p1 p2 p3.
    """
    estimator = StatevectorEstimator()
    bound = ansatz.assign_parameters({p: v for p, v in zip(param_list, param_vals)})
    job = estimator.run([(bound, HAMILTONIAN)])
    return float(job.result()[0].data.evs.real)


def _parity(counts, n_shots):
    """
    ZZ parity estimator from bitstring counts.
    Works for any term measured in the appropriate rotated basis.
    <P> = (N_00 - N_01 - N_10 + N_11) / N
    """
    n00 = counts.get("00", 0)
    n01 = counts.get("01", 0)
    n10 = counts.get("10", 0)
    n11 = counts.get("11", 0)
    return (n00 - n01 - n10 + n11) / n_shots


def _build_base_circuit(param_vals):
    """Ansatz circuit with concrete parameter values, no measurement."""
    t0, t1, t2, t3, p0, p1, p2, p3 = param_vals
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.ry(t0, 0)
    qc.rz(p0, 0)
    qc.ry(t1, 1)
    qc.rz(p1, 1)
    qc.cx(0, 1)
    qc.ry(t2, 0)
    qc.rz(p2, 0)
    qc.ry(t3, 1)
    qc.rz(p3, 1)
    return qc


def _circuit_zz(param_vals):
    """Z⊗Z measurement: computational basis."""
    qc = _build_base_circuit(param_vals)
    qc.measure([0, 1], [0, 1])
    return qc


def _circuit_xx(param_vals):
    """
    X⊗X measurement: rotate both qubits by H to diagonalize X,
    then measure in computational basis.
    X = H·Z·H  =>  <X⊗X> = parity in H-rotated basis.
    """
    qc = _build_base_circuit(param_vals)
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc


def _circuit_yy(param_vals):
    """
    Y⊗Y measurement: rotate both qubits by Sdg·H to diagonalize Y,
    then measure in computational basis.
    Y = H·S·Z·Sdg·H  =>  Sdg·H diagonalizes Y.
    """
    qc = _build_base_circuit(param_vals)
    qc.sdg(0)
    qc.h(0)
    qc.sdg(1)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc


def sampled_expectation(param_vals, simulator, n_shots=2000, seed=None):
    """
    <XX + YY + ZZ> from shot-based sampling using three circuits,
    one per Pauli term, each measuring in the appropriate rotated basis.
    """
    circuits = [
        _circuit_xx(param_vals),
        _circuit_yy(param_vals),
        _circuit_zz(param_vals),
    ]
    compiled = transpile(circuits, simulator)
    results = simulator.run(compiled, shots=n_shots, seed_simulator=seed).result()

    exp_xx = _parity(results.get_counts(0), n_shots)
    exp_yy = _parity(results.get_counts(1), n_shots)
    exp_zz = _parity(results.get_counts(2), n_shots)

    return exp_xx + exp_yy + exp_zz


# =============================================================================
# VQE optimization
# =============================================================================


def run_vqe(ansatz, param_list, x0):
    """
    COBYLA minimization of <XX + YY + ZZ>.
    x0: initial parameter vector of length 8.
    Returns optimal params, optimal energy, iteration history.
    """
    history = []

    def objective(x):
        val = exact_expectation(x, ansatz, param_list)
        history.append((x.copy(), val))
        return val

    result = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": 800, "rhobeg": 0.5},
    )

    return result.x, result.fun, history


# =============================================================================
# Plotting
# =============================================================================


def plot_parameter_sweep(ansatz, param_list, optimal_params):
    """
    Figure 1: 1D sweep of t0 with all other parameters fixed at optimum.
    The Heisenberg landscape is richer than the non-interacting case;
    the sweep reveals whether the optimizer sits in a genuine minimum.
    """
    theta_range = np.linspace(0, 2 * np.pi, 200)
    energies = []
    base = optimal_params.copy()

    for t in theta_range:
        base[0] = t
        energies.append(exact_expectation(base, ansatz, param_list))

    plt.figure(1, figsize=(8, 4))
    plt.plot(
        theta_range, energies, linewidth=1.5, label=r"$\langle H \rangle$ vs $\theta_0$"
    )
    plt.axhline(
        TRUE_GROUND_ENERGY,
        color="red",
        linestyle="--",
        label=f"Ground state E = {TRUE_GROUND_ENERGY}",
    )
    plt.axvline(
        optimal_params[0] % (2 * np.pi),
        color="black",
        linestyle=":",
        label=f"Optimal $\\theta_0$ = {optimal_params[0] % (2 * np.pi):.3f} rad",
    )
    plt.xlabel(r"$\theta_0$ [radians]")
    plt.ylabel(r"$\langle XX + YY + ZZ \rangle$")
    plt.title(
        r"Energy landscape slice: H = XX + YY + ZZ"
        "\n(all other parameters fixed at optimum)"
    )
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("vqe_Heis_landscape.png", dpi=150)


def plot_convergence(history, optimal_energy):
    """Figure 2: <H> at each COBYLA iteration."""
    energies = [h[1] for h in history]

    plt.figure(2, figsize=(8, 4))
    plt.plot(energies, "-o", markersize=3, linewidth=1)
    plt.axhline(
        TRUE_GROUND_ENERGY,
        color="red",
        linestyle="--",
        label=f"True ground state: {TRUE_GROUND_ENERGY:.4f}",
    )
    plt.xlabel("Optimizer iteration")
    plt.ylabel(r"$\langle H \rangle$")
    plt.title(
        "VQE Convergence: H = XX + YY + ZZ\n"
        f"Iterations: {len(history)}    "
        f"Final \u27e8H\u27e9: {optimal_energy:.6f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("vqe_Heis_convergence.png", dpi=150)


def plot_pauli_terms(param_vals, ansatz, param_list, label, fig_num):
    """
    Figure 3 or 5: Bar chart of individual Pauli term contributions
    <XX>, <YY>, <ZZ> at a given parameter point.
    Pedagogically useful: shows how each term contributes to the total.
    For the singlet: <XX> = <YY> = <ZZ> = -1, summing to -3.
    """
    estimator = StatevectorEstimator()
    bound = ansatz.assign_parameters({p: v for p, v in zip(param_list, param_vals)})

    terms = [
        ("XX", SparsePauliOp.from_list([("XX", 1.0)])),
        ("YY", SparsePauliOp.from_list([("YY", 1.0)])),
        ("ZZ", SparsePauliOp.from_list([("ZZ", 1.0)])),
    ]

    values = []
    for name, op in terms:
        job = estimator.run([(bound, op)])
        values.append(float(job.result()[0].data.evs.real))

    colors = ["steelblue", "seagreen", "tomato"]
    plt.figure(fig_num, figsize=(6, 4))
    bars = plt.bar([t[0] for t in terms], values, color=colors, width=0.4)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(
        -1.0,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Each term = -1 at singlet",
    )
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.12,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.ylabel(r"$\langle \cdot \rangle$")
    plt.ylim(-1.5, 1.5)
    plt.title(
        f"Pauli term contributions at {label}\nTotal \u27e8H\u27e9 = {sum(values):.4f}"
    )
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"vqe_Heis_pauliterms_{label.lower().replace(' ', '_')}.png", dpi=150)


# --- Histogram infrastructure ---


@dataclass
class HistogramData:
    """Z-basis bitstring probabilities at a given parameter point."""

    p_exact: dict
    p_ideal_mean: dict
    p_ideal_std: dict
    p_noisy_mean: dict
    p_noisy_std: dict
    p_fake_mean: dict
    p_fake_std: dict
    n_shots: int
    n_reps: int


BITSTRINGS = ["00", "01", "10", "11"]


def _exact_probs(param_vals, ansatz, param_list):
    """Exact Z-basis probabilities from statevector."""
    from qiskit.quantum_info import Statevector

    bound = ansatz.assign_parameters({p: v for p, v in zip(param_list, param_vals)})
    probs = Statevector(bound).probabilities_dict()
    return {bs: probs.get(bs, 0.0) for bs in BITSTRINGS}


def _sampled_probs(param_vals, simulator, n_shots, n_reps, rng):
    """
    Repeated shot-based Z-basis probabilities.
    Returns mean and std dicts over n_reps repetitions.
    Note: Z-basis measurement only — does not capture the relative
    phase between |01⟩ and |10⟩ that distinguishes singlet from triplet.
    """
    qc = _build_base_circuit(param_vals)
    qc.measure([0, 1], [0, 1])

    compiled = transpile(qc, simulator)
    runs = {bs: [] for bs in BITSTRINGS}

    for rep_seed in rng.integers(2**31, size=n_reps):
        counts = (
            simulator.run(compiled, shots=n_shots, seed_simulator=int(rep_seed))
            .result()
            .get_counts()
        )
        total = sum(counts.values())
        for bs in BITSTRINGS:
            runs[bs].append(counts.get(bs, 0) / total)

    means = {bs: float(np.mean(runs[bs])) for bs in BITSTRINGS}
    stds = {bs: float(np.std(runs[bs])) for bs in BITSTRINGS}
    return means, stds


def build_histogram_data(
    param_vals,
    ideal_sim,
    noisy_sim,
    fake_sim,
    ansatz,
    param_list,
    n_shots=2000,
    n_reps=20,
    seed=None,
):
    rng = np.random.default_rng(seed)
    p_exact = _exact_probs(param_vals, ansatz, param_list)
    p_ideal_mean, p_ideal_std = _sampled_probs(
        param_vals, ideal_sim, n_shots, n_reps, rng
    )
    p_noisy_mean, p_noisy_std = _sampled_probs(
        param_vals, noisy_sim, n_shots, n_reps, rng
    )
    p_fake_mean, p_fake_std = _sampled_probs(
        param_vals, fake_sim, n_shots, n_reps, rng
    )

    return HistogramData(
        p_exact,
        p_ideal_mean,
        p_ideal_std,
        p_noisy_mean,
        p_noisy_std,
        p_fake_mean,
        p_fake_std,
        n_shots,
        n_reps,
    )


def draw_histogram(
    fig_num,
    param_vals,
    title_label,
    savefile,
    ideal_sim,
    noisy_sim,
    fake_sim,
    ansatz,
    param_list,
):
    """
    Grouped bar histogram of Z-basis bitstring probabilities.
    At the optimal point (singlet) the exact histogram shows
    |01⟩ ≈ 0.5 and |10⟩ ≈ 0.5, with |00⟩ = |11⟩ = 0.
    The relative phase between |01⟩ and |10⟩ is invisible here —
    both the singlet (|01⟩ - |10⟩)/√2 and the triplet (|01⟩ + |10⟩)/√2
    produce identical Z-basis histograms. The Pauli term plot resolves this.
    """
    d = build_histogram_data(
        param_vals, ideal_sim, noisy_sim, fake_sim, ansatz, param_list
    )

    x = np.arange(len(BITSTRINGS))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    bar_specs = [
        ("Exact (Statevector)", "steelblue", d.p_exact, None),
        (f"Ideal ({d.n_reps} runs)", "seagreen", d.p_ideal_mean, d.p_ideal_std),
        (f"Noisy ({d.n_reps} runs)", "tomato", d.p_noisy_mean, d.p_noisy_std),
        (
            f"FakeSherbrooke ({d.n_reps} runs)",
            "mediumpurple",
            d.p_fake_mean,
            d.p_fake_std,
        ),
    ]

    fig, ax = plt.subplots(num=fig_num, figsize=(10, 5))

    for i, (label, color, means, stds) in enumerate(bar_specs):
        heights = [means[bs] for bs in BITSTRINGS]
        yerr = [stds[bs] for bs in BITSTRINGS] if stds else None
        bars = ax.bar(
            x + offsets[i],
            heights,
            width,
            label=label,
            color=color,
            yerr=yerr,
            capsize=4 if yerr else 0,
            error_kw={"elinewidth": 1.5} if yerr else {},
        )
        for bar, bs in zip(bars, BITSTRINGS):
            h = bar.get_height()
            ax.annotate(
                f"{means[bs]:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 18 if stds else 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            if stds:
                ax.annotate(
                    f"\u00b1{stds[bs]:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h + stds[bs]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="dimgray",
                    style="italic",
                )

    ax.text(
        0.98,
        0.98,
        f"Error bars = std over\n{d.n_reps} reps \u00d7 {d.n_shots} shots\n"
        "Note: Z-basis only — phase between\n|01\u27e9 and |10\u27e9 not visible here",
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"|{bs}\u27e9" for bs in BITSTRINGS], fontsize=12)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.25)
    ax.set_title(
        f"Z-basis measurement outcomes at {title_label}\n"
        f"(mean \u00b1 std over {d.n_reps} reps \u00d7 {d.n_shots} shots)"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(savefile, dpi=150)


# =============================================================================
# Main
# =============================================================================


def main():
    print("\n" + "=" * 60)
    print("VQE: Isotropic Heisenberg model H = XX + YY + ZZ")
    print("=" * 60)
    print("Spectrum:")
    print("  |00\u27e9                    E = +1  (triplet)")
    print("  |11\u27e9                    E = +1  (triplet)")
    print("  (|01\u27e9 + |10\u27e9)/\u221a2      E = +1  (triplet)")
    print("  (|01\u27e9 - |10\u27e9)/\u221a2      E = -3  (singlet) <- ground state")
    print(f"\nTrue ground state energy : {TRUE_GROUND_ENERGY}")
    print("Ground state is maximally entangled: CNOT in ansatz is essential.")

    # Build components
    ansatz, param_list = build_ansatz()
    ideal_sim = build_ideal_simulator()
    noisy_sim = build_noisy_simulator()
    fake_sim = build_fake_simulator()

    print("\nAnsatz circuit:")
    print(ansatz.draw(output="text"))

    # Initial parameters
    rng = np.random.default_rng(42)
    x0 = rng.uniform(0.1, 0.5, size=8)

    # Sanity check at x0
    e0 = exact_expectation(x0, ansatz, param_list)
    e0_ideal = sampled_expectation(x0, ideal_sim)
    e0_noisy = sampled_expectation(x0, noisy_sim)
    e0_fake = sampled_expectation(x0, fake_sim)

    print("\nAt initial parameters:")
    print(f"  StatevectorEstimator : {e0:.6f}")
    print(f"  Ideal sampled        : {e0_ideal:.6f}")
    print(f"  Noisy sampled        : {e0_noisy:.6f}")
    print(f"  FakeSherbrooke       : {e0_fake:.6f}")

    # VQE
    print("\nRunning VQE (COBYLA) ...")
    opt_params, opt_energy, history = run_vqe(ansatz, param_list, x0)

    print("\nVQE Result")
    print(f"  Iterations     : {len(history)}")
    print(
        f"  Optimal \u27e8H\u27e9    : {opt_energy:.8f}  (true: {
            TRUE_GROUND_ENERGY:.8f})"
    )
    print(f"  Error          : {abs(opt_energy - TRUE_GROUND_ENERGY):.2e}")
    print("  Parameters     :")
    labels = ["t0", "t1", "t2", "t3", "p0", "p1", "p2", "p3"]
    for lbl, val in zip(labels, opt_params):
        print(f"    {lbl} = {val:.6f} rad")

    # Verify at optimal point
    e_sv = exact_expectation(opt_params, ansatz, param_list)
    e_ideal = sampled_expectation(opt_params, ideal_sim)
    e_noisy = sampled_expectation(opt_params, noisy_sim)
    e_fake = sampled_expectation(opt_params, fake_sim)

    print("\n<XX + YY + ZZ> at optimal params:")
    print(f"  StatevectorEstimator : {e_sv:.6f}")
    print(f"  Ideal sampled        : {e_ideal:.6f}")
    print(f"  Noisy sampled        : {e_noisy:.6f}")
    print(f"  FakeSherbrooke       : {e_fake:.6f}")

    # Plots
    plot_parameter_sweep(ansatz, param_list, opt_params)
    plot_convergence(history, opt_energy)
    plot_pauli_terms(x0, ansatz, param_list, "Initial Parameters", 3)
    plot_pauli_terms(opt_params, ansatz, param_list, "Optimal Parameters", 4)

    draw_histogram(
        5,
        x0,
        "Initial Parameters",
        "vqe_Heis_histogram_initial.png",
        ideal_sim,
        noisy_sim,
        fake_sim,
        ansatz,
        param_list,
    )
    draw_histogram(
        6,
        opt_params,
        "Optimal Parameters",
        "vqe_Heis_histogram_optimal.png",
        ideal_sim,
        noisy_sim,
        fake_sim,
        ansatz,
        param_list,
    )

    plt.show()


if __name__ == "__main__":
    main()
