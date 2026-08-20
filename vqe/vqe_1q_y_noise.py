#!/usr/bin/env python3

# VQE demonstration: single-qubit Hamiltonian minimization using Qiskit.
# Hamiltonian: H = Y (Rx ansatz, ground state |−i⟩).
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
# Circuit and Hamiltonian setup
# =============================================================================


def build_ansatz():
    theta = Parameter("θ")
    ansatz = QuantumCircuit(1)
    ansatz.rx(theta, 0)
    return ansatz


def build_hamiltonian():
    # H = Y_0
    return SparsePauliOp("Y")


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

    # Compose depolarizing and thermal relaxation into a single error channel
    dep_error = depolarizing_error(0.01, 1)
    thermal_error = thermal_relaxation_error(t1, t2, gate_time)
    combined_error = dep_error.compose(thermal_error)
    noise_model.add_all_qubit_quantum_error(combined_error, ["rx", "ry", "h", "sdg"])

    # Readout error: 2% chance of flipping 0->1 or 1->0
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


def exact_expectation(angle, ansatz, hamiltonian):
    """Deterministic expectation via full statevector contraction."""
    estimator = StatevectorEstimator()
    bound = ansatz.assign_parameters([angle])
    job = estimator.run([(bound, hamiltonian)])
    return job.result()[0].data.evs.real


def sampled_expectation(angle, simulator, n_samples=1000):
    """Manual Y expectation by rotating into Y eigenbasis before measuring."""
    qc = QuantumCircuit(1, 1)
    qc.rx(angle, 0)
    qc.sdg(0)
    qc.h(0)
    qc.measure(0, 0)
    compiled = transpile(qc, simulator)
    counts = simulator.run(compiled, shots=n_samples).result().get_counts()
    shots_0 = counts.get("0", 0)
    shots_1 = counts.get("1", 0)
    return (shots_0 - shots_1) / n_samples


# =============================================================================
# VQE optimization
# =============================================================================


def run_vqe(ansatz, hamiltonian, theta_init=1.0):
    """Run VQE using COBYLA optimizer. Returns optimal angle, energy, history."""
    history = []

    def objective(x):
        """Function that the optimizer minimizes."""
        val = exact_expectation(x[0], ansatz, hamiltonian)
        history.append((x[0], val))
        return val

    result = minimize(
        objective,
        np.array([theta_init]),
        method="COBYLA",
        options={"maxiter": 200, "rhobeg": 0.5},
    )

    return result.x[0], result.fun, history


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


def plot_landscape(
    angle_range, exact, sampled, noisy, fake, history, optimal_angle, optimal_energy
):
    """Figure 1: energy landscape with optimizer path overlaid."""
    plt.figure(1, figsize=(9, 5))
    plt.plot(angle_range, exact, linewidth=2, label="Exact (StatevectorEstimator)")
    plt.plot(
        angle_range, sampled, "-o", markersize=5, label="Sampled (Ideal simulator)"
    )
    plt.plot(angle_range, noisy, "-s", markersize=5, label="Sampled (Noisy simulator)")
    plt.plot(angle_range, fake, "-^", markersize=5, label="Sampled (FakeManilaV2)")

    hist_angles = [h[0] % (2 * np.pi) for h in history]
    hist_values = [h[1] for h in history]
    plt.scatter(
        hist_angles, hist_values, color="red", zorder=5, s=40, label="Optimizer steps"
    )
    plt.scatter(
        optimal_angle % (2 * np.pi),
        optimal_energy,
        color="black",
        zorder=6,
        marker="D",
        s=75,
        label=f"Minimum: θ={optimal_angle:.3f}, ⟨Y⟩={optimal_energy:.3f}",
    )

    plt.xlabel("Angle [radians]")
    plt.ylabel(r"$\langle Y \rangle$")
    plt.title("VQE Energy Landscape: Single Rx Ansatz")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("vqe_expectation.png", dpi=150)


def plot_convergence(history, optimal_energy):
    """Figure 2: ⟨Y⟩ value at each optimizer iteration."""
    plt.figure(2, figsize=(7, 4))
    plt.plot([h[1] for h in history], "-o", markersize=4)
    plt.axhline(y=-1.0, color="red", linestyle="--", label="True ground state")
    plt.xlabel("Optimizer iteration")
    plt.ylabel(r"$\langle Y \rangle$")
    plt.title(
        f"VQE Convergence\nIterations: {len(history)}   Final ⟨Y⟩: {optimal_energy:.6f}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("vqe_convergence.png", dpi=150)


def build_histogram_data(
    angle, ideal_simulator, noisy_simulator, fake_simulator, ansatz, hamiltonian
):
    """Compute all probabilities and statistics needed for histogram figures."""
    n_shots, n_repetitions = 1000, 20

    # For Y: <Y> = P(+i) - P(-i), so P(-i) = (1 - <Y>) / 2
    exact_val = exact_expectation(angle, ansatz, hamiltonian)
    p1_exact = (1 - exact_val) / 2
    p0_exact = 1 - p1_exact

    # Sdg + H rotates Y eigenbasis to Z eigenbasis
    qc = QuantumCircuit(1, 1)
    qc.rx(angle, 0)
    qc.sdg(0)
    qc.h(0)
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
    angle,
    title_label,
    savefile,
    ideal_simulator,
    noisy_simulator,
    fake_simulator,
    ansatz,
    hamiltonian,
):
    """Draw a grouped bar histogram for a given angle."""
    d = build_histogram_data(
        angle, ideal_simulator, noisy_simulator, fake_simulator, ansatz, hamiltonian
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
    ax.set_xticklabels(["|+i⟩", "|−i⟩"], fontsize=13)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.3)
    ax.set_title(
        f"Measurement Outcomes at {title_label} θ = {angle:.4f} rad\n"
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
    ansatz = build_ansatz()
    hamiltonian = build_hamiltonian()
    ideal_simulator = build_ideal_simulator()
    noisy_simulator = build_noisy_simulator()
    fake_simulator = build_fake_simulator()

    print("\nAnsatz circuit:")
    print(ansatz.draw(output="text"))

    # Single angle check
    angle = 1.2
    print(f"\nAt angle = {angle} rad:")
    print(
        f"  StatevectorEstimator : {exact_expectation(angle, ansatz, hamiltonian):.6f}"
    )
    print(f"  Ideal sampled        : {sampled_expectation(angle, ideal_simulator):.6f}")
    print(f"  Noisy sampled        : {sampled_expectation(angle, noisy_simulator):.6f}")
    print(f"  FakeManilaV2         : {sampled_expectation(angle, fake_simulator):.6f}")

    # VQE optimization
    theta_init = 1.0
    optimal_angle, optimal_energy, history = run_vqe(
        ansatz, hamiltonian, theta_init=theta_init
    )

    print("\nVQE Optimization Result")
    print(f"  Optimal angle  : {optimal_angle:.6f} rad")
    print(f"  Optimal ⟨H⟩    : {optimal_energy:.6f}     (expected -1.0)")
    print(f"  Iterations     : {len(history)}")

    # Sweep for landscape
    angle_range = np.linspace(0.0, 2.0 * np.pi, 20)
    exact = [exact_expectation(a, ansatz, hamiltonian) for a in angle_range]
    sampled = [sampled_expectation(a, ideal_simulator) for a in angle_range]
    noisy = [sampled_expectation(a, noisy_simulator) for a in angle_range]
    fake = [sampled_expectation(a, fake_simulator) for a in angle_range]

    # Plots
    plot_landscape(
        angle_range, exact, sampled, noisy, fake, history, optimal_angle, optimal_energy
    )
    plot_convergence(history, optimal_energy)
    draw_histogram(
        3,
        theta_init,
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
        optimal_angle,
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
