#!/usr/bin/env python3

# VQE demonstration: single-qubit Hamiltonian minimization using Qiskit.
# Hamiltonian: H = cos(alpha)*X + sin(alpha)*Z
# Ground state energy: -1 for all alpha.
# Two-parameter Rz·Ry ansatz covering the full Bloch sphere.
# Compares exact (StatevectorEstimator), ideal sampled, manually noisy,
# and fake backend (FakeManilaV2) sampled expectations.
# Optimization via COBYLA (scipy).

# --- Qiskit imports ---
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
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp


# =============================================================================
# Hamiltonian parameter — change alpha to explore different landscapes
# =============================================================================

ALPHA = np.pi / 4  # 45 degrees: equal mix of X and Z


# =============================================================================
# Circuit and Hamiltonian setup
# =============================================================================


def build_ansatz():
    """Two-parameter Rz·Ry ansatz covering the full Bloch sphere."""
    theta = Parameter("θ")
    phi = Parameter("φ")
    ansatz = QuantumCircuit(1)
    ansatz.ry(theta, 0)
    ansatz.rz(phi, 0)
    return ansatz


def build_hamiltonian(alpha):
    """H = cos(alpha)*X + sin(alpha)*Z"""
    return SparsePauliOp.from_list(
        [
            ("X", np.cos(alpha)),
            ("Z", np.sin(alpha)),
        ]
    )


# =============================================================================
# Simulator setup
# =============================================================================


def build_ideal_simulator():
    """Standard noiseless AerSimulator."""
    return AerSimulator()


def build_noisy_simulator():
    """AerSimulator with a manually constructed noise model
    emulating a typical superconducting qubit device."""
    noise_model = NoiseModel()

    # T1 = 100 us, T2 = 80 us, gate time = 50 ns
    t1, t2, gate_time = 100e-6, 80e-6, 50e-9

    dep_error = depolarizing_error(0.01, 1)
    thermal_error = thermal_relaxation_error(t1, t2, gate_time)
    combined_error = dep_error.compose(thermal_error)
    noise_model.add_all_qubit_quantum_error(
        combined_error, ["rx", "ry", "rz", "h", "sdg"]
    )

    readout_error = ReadoutError([[0.98, 0.02], [0.02, 0.98]])
    noise_model.add_all_qubit_readout_error(readout_error)

    return AerSimulator(noise_model=noise_model)


def build_fake_simulator():
    """AerSimulator built from a real IBM QPU snapshot (FakeManilaV2)."""
    backend = FakeManilaV2()
    return AerSimulator.from_backend(backend)


# =============================================================================
# Expectation value functions
# =============================================================================


def exact_expectation(params, ansatz, hamiltonian):
    """Deterministic expectation via full statevector contraction.
    params = [theta, phi]"""
    estimator = StatevectorEstimator()
    bound = ansatz.assign_parameters(params)
    job = estimator.run([(bound, hamiltonian)])
    return job.result()[0].data.evs.real


def sampled_expectation(params, simulator, alpha, n_samples=1000):
    """Sampled <H> = cos(alpha)*<X> + sin(alpha)*<Z> via two circuit runs."""
    theta, phi = params

    # --- Measure <Z> ---
    qc_z = QuantumCircuit(1, 1)
    qc_z.ry(theta, 0)
    qc_z.rz(phi, 0)
    qc_z.measure(0, 0)
    compiled_z = transpile(qc_z, simulator)
    counts_z = simulator.run(compiled_z, shots=n_samples).result().get_counts()
    exp_z = (counts_z.get("0", 0) - counts_z.get("1", 0)) / n_samples

    # --- Measure <X> (H rotates X eigenbasis to Z eigenbasis) ---
    qc_x = QuantumCircuit(1, 1)
    qc_x.ry(theta, 0)
    qc_x.rz(phi, 0)
    qc_x.h(0)
    qc_x.measure(0, 0)
    compiled_x = transpile(qc_x, simulator)
    counts_x = simulator.run(compiled_x, shots=n_samples).result().get_counts()
    exp_x = (counts_x.get("0", 0) - counts_x.get("1", 0)) / n_samples

    return np.cos(alpha) * exp_x + np.sin(alpha) * exp_z


# =============================================================================
# VQE optimization
# =============================================================================


def run_vqe(ansatz, hamiltonian, theta_init=1.0, phi_init=0.5):
    """Run VQE using COBYLA optimizer. Returns optimal params, energy, history."""
    history = []

    def objective(x):
        val = exact_expectation(x, ansatz, hamiltonian)
        history.append((x.copy(), val))
        return val

    result = minimize(
        objective,
        np.array([theta_init, phi_init]),
        method="COBYLA",
        options={"maxiter": 400, "rhobeg": 0.5},
    )

    return result.x, result.fun, history


# =============================================================================
# Plotting functions
# =============================================================================


@dataclass
class HistogramData:
    """Container for histogram computation results."""

    p0_exact: float
    p1_exact: float
    p0_ideal_mean: float
    p0_ideal_std: float
    p1_ideal_mean: float
    p1_ideal_std: float
    p0_noisy_mean: float
    p0_noisy_std: float
    p1_noisy_mean: float
    p1_noisy_std: float
    p0_fake_mean: float
    p0_fake_std: float
    p1_fake_mean: float
    p1_fake_std: float
    n_shots: int
    n_repetitions: int


def plot_landscape(history, optimal_params, optimal_energy, ansatz, hamiltonian, alpha):
    """Figure 1: 2D heatmap of <H> over (theta, phi) with optimizer path."""
    n_points = 30
    theta_range = np.linspace(0, 2 * np.pi, n_points)
    phi_range = np.linspace(0, 2 * np.pi, n_points)

    energy_grid = np.zeros((n_points, n_points))
    for i, theta in enumerate(theta_range):
        for j, phi in enumerate(phi_range):
            energy_grid[i, j] = exact_expectation([theta, phi], ansatz, hamiltonian)

    path_theta = [h[0][0] % (2 * np.pi) for h in history]
    path_phi = [h[0][1] % (2 * np.pi) for h in history]

    plt.figure(1, figsize=(9, 6))
    cp = plt.contourf(phi_range, theta_range, energy_grid, levels=30, cmap="RdYlBu_r")
    plt.colorbar(cp, label=r"$\langle H \rangle$")
    plt.plot(
        path_phi,
        path_theta,
        "w-o",
        markersize=3,
        linewidth=1,
        alpha=0.7,
        label="Optimizer path",
    )
    plt.scatter(
        path_phi[0],
        path_theta[0],
        color="white",
        zorder=6,
        s=80,
        marker="o",
        label="Initial guess",
    )
    plt.scatter(
        optimal_params[1] % (2 * np.pi),
        optimal_params[0] % (2 * np.pi),
        color="gold",
        zorder=7,
        marker="D",
        s=100,
        label=f"Minimum: ⟨H⟩={optimal_energy:.4f}",
    )

    plt.xlabel("φ [radians]")
    plt.ylabel("θ [radians]")
    plt.title(f"VQE Energy Landscape: H = cos(α)X + sin(α)Z,  α = {alpha:.4f} rad")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("vqe_expectation.png", dpi=150)


def plot_convergence(history, optimal_energy):
    """Figure 2: <H> value at each optimizer iteration."""
    plt.figure(2, figsize=(7, 4))
    plt.plot([h[1] for h in history], "-o", markersize=4)
    plt.axhline(y=-1.0, color="red", linestyle="--", label="True ground state: -1")
    plt.xlabel("Optimizer iteration")
    plt.ylabel(r"$\langle H \rangle$")
    plt.title(
        f"VQE Convergence\nIterations: {len(history)}   Final ⟨H⟩: {optimal_energy:.6f}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("vqe_convergence.png", dpi=150)


def build_histogram_data(
    params, ideal_simulator, noisy_simulator, fake_simulator, ansatz, hamiltonian
):
    """Compute probabilities and statistics for histogram figures (Z basis)."""
    n_shots, n_repetitions = 1000, 20
    theta, phi = params

    # Exact Z-basis probabilities from statevector
    estimator = StatevectorEstimator()
    bound = ansatz.assign_parameters(params)
    job = estimator.run([(bound, SparsePauliOp("Z"))])
    exp_z = job.result()[0].data.evs.real
    p1_exact = (1 - exp_z) / 2
    p0_exact = 1 - p1_exact

    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.measure(0, 0)

    def repeated_probs(simulator):
        p0s, p1s = [], []
        compiled = transpile(qc, simulator)
        for _ in range(n_repetitions):
            counts = simulator.run(compiled, shots=n_shots).result().get_counts()
            counts.setdefault("0", 0)
            counts.setdefault("1", 0)
            total = sum(counts.values())
            p0s.append(counts["0"] / total)
            p1s.append(counts["1"] / total)
        return np.mean(p0s), np.std(p0s), np.mean(p1s), np.std(p1s)

    p0_im, p0_is, p1_im, p1_is = repeated_probs(ideal_simulator)
    p0_nm, p0_ns, p1_nm, p1_ns = repeated_probs(noisy_simulator)
    p0_fm, p0_fs, p1_fm, p1_fs = repeated_probs(fake_simulator)

    return HistogramData(
        p0_exact,
        p1_exact,
        p0_im,
        p0_is,
        p1_im,
        p1_is,
        p0_nm,
        p0_ns,
        p1_nm,
        p1_ns,
        p0_fm,
        p0_fs,
        p1_fm,
        p1_fs,
        n_shots,
        n_repetitions,
    )


def draw_histogram(
    fig_number,
    params,
    title_label,
    savefile,
    ideal_simulator,
    noisy_simulator,
    fake_simulator,
    ansatz,
    hamiltonian,
):
    """Draw a grouped bar histogram for a given set of parameters."""
    d = build_histogram_data(
        params, ideal_simulator, noisy_simulator, fake_simulator, ansatz, hamiltonian
    )

    x, width = np.array([0, 1]), 0.2
    offset = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(num=int(fig_number), figsize=(9, 5))

    bar_specs = [
        ([d.p0_exact, d.p1_exact], None, "Exact (StatevectorEstimator)", "steelblue"),
        (
            [d.p0_ideal_mean, d.p1_ideal_mean],
            [d.p0_ideal_std, d.p1_ideal_std],
            f"Sampled (Ideal, {d.n_repetitions} runs)",
            "seagreen",
        ),
        (
            [d.p0_noisy_mean, d.p1_noisy_mean],
            [d.p0_noisy_std, d.p1_noisy_std],
            f"Sampled (Noisy, {d.n_repetitions} runs)",
            "tomato",
        ),
        (
            [d.p0_fake_mean, d.p1_fake_mean],
            [d.p0_fake_std, d.p1_fake_std],
            f"Sampled (FakeManilaV2, {d.n_repetitions} runs)",
            "mediumpurple",
        ),
    ]

    mean_labels = [
        [f"{d.p0_exact * 100:.1f}%", f"{d.p1_exact * 100:.1f}%"],
        [f"{d.p0_ideal_mean:.3f}", f"{d.p1_ideal_mean:.3f}"],
        [f"{d.p0_noisy_mean:.3f}", f"{d.p1_noisy_mean:.3f}"],
        [f"{d.p0_fake_mean:.3f}", f"{d.p1_fake_mean:.3f}"],
    ]
    stds_list = [
        [0, 0],
        [d.p0_ideal_std, d.p1_ideal_std],
        [d.p0_noisy_std, d.p1_noisy_std],
        [d.p0_fake_std, d.p1_fake_std],
    ]

    for i, (heights, yerr, label, color) in enumerate(bar_specs):
        bars = ax.bar(
            x + offset[i],
            heights,
            width,
            label=label,
            color=color,
            yerr=yerr,
            capsize=4 if yerr else 0,
            error_kw={"elinewidth": 1.5} if yerr else {},
        )
        for bar, mean_label, std in zip(bars, mean_labels[i], stds_list[i]):
            height = bar.get_height()
            ax.annotate(
                mean_label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 18 if std > 0 else 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            if std > 0:
                ax.annotate(
                    f"±{std:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height + std),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="dimgray",
                    style="italic",
                )

    ax.text(
        0.98,
        0.98,
        f"Error bars = std over\n{d.n_repetitions} repetitions of {d.n_shots} shots",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["|0⟩", "|1⟩"], fontsize=13)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.3)
    ax.set_title(
        f"Z-basis Measurement Outcomes at {title_label}\n"
        f"θ = {params[0]:.4f} rad, φ = {params[1]:.4f} rad\n"
        f"(mean ± std over {d.n_repetitions} repetitions of {d.n_shots} shots)"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(savefile, dpi=150)


# =============================================================================
# Main
# =============================================================================


def main():
    # Setup
    alpha = ALPHA
    ansatz = build_ansatz()
    hamiltonian = build_hamiltonian(alpha)
    ideal_simulator = build_ideal_simulator()
    noisy_simulator = build_noisy_simulator()
    fake_simulator = build_fake_simulator()

    print(
        f"\nHamiltonian: H = cos(α)X + sin(α)Z,  α = {alpha:.4f} rad ({
            np.degrees(alpha):.1f}°)"
    )
    print("True ground state energy: -1.000000")
    print(
        f"Optimal angle (analytic): θ* = {-alpha - np.pi / 2:.4f} rad, φ* = 0.0000 rad"
    )

    print("\nAnsatz circuit:")
    print(ansatz.draw(output="text"))

    # Single parameter check
    params_check = [1.0, 0.5]
    print(f"\nAt θ = {params_check[0]}, φ = {params_check[1]} rad:")
    print(
        f"  StatevectorEstimator : {
            exact_expectation(params_check, ansatz, hamiltonian):.6f}"
    )
    print(
        f"  Ideal sampled        : {
            sampled_expectation(params_check, ideal_simulator, alpha):.6f}"
    )
    print(
        f"  Noisy sampled        : {
            sampled_expectation(params_check, noisy_simulator, alpha):.6f}"
    )
    print(
        f"  FakeManilaV2         : {
            sampled_expectation(params_check, fake_simulator, alpha):.6f}"
    )

    # VQE optimization
    theta_init, phi_init = 1.0, 0.5
    optimal_params, optimal_energy, history = run_vqe(
        ansatz, hamiltonian, theta_init=theta_init, phi_init=phi_init
    )

    print("\nVQE Optimization Result")
    print(f"  Optimal θ      : {optimal_params[0]:.6f} rad")
    print(f"  Optimal φ      : {optimal_params[1]:.6f} rad")
    print(f"  Optimal ⟨H⟩    : {optimal_energy:.6f}     (expected -1.0)")
    print(f"  Iterations     : {len(history)}")

    # Plots
    plot_landscape(history, optimal_params, optimal_energy, ansatz, hamiltonian, alpha)
    plot_convergence(history, optimal_energy)
    draw_histogram(
        3,
        [theta_init, phi_init],
        "Initial Guess",
        "vqe_histogram_initial.png",
        ideal_simulator,
        noisy_simulator,
        fake_simulator,
        ansatz,
        hamiltonian,
    )
    draw_histogram(
        4,
        optimal_params,
        "Optimal",
        "vqe_histogram_optimal.png",
        ideal_simulator,
        noisy_simulator,
        fake_simulator,
        ansatz,
        hamiltonian,
    )

    plt.show()


if __name__ == "__main__":
    main()
