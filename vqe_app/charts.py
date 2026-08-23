"""Matplotlib figures built from the plain dicts vqe_app.scenarios returns.

Matplotlib rather than a plotting library with more interactivity: the vqe/
scripts already depend on it, so this adds nothing to the install, and the
interactivity that matters here is in the parameter widgets, not the axes.

Colours are fixed for the dark theme pinned in .streamlit/config.toml.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

BACKGROUND = "#0e1117"
FOREGROUND = "#fafafa"
MUTED = "#8b949e"
GRID = "#2c333d"

COLORS = {
    "exact": "#4c8dd8",
    "ideal": "#3fa16b",
    "noisy": "#e0603f",
    "fake": "#9b7fd4",
    "target": "#e05252",
}

SERIES = [
    ("exact", "Exact (statevector)", COLORS["exact"]),
    ("ideal", "Sampled — ideal", COLORS["ideal"]),
    ("noisy", "Sampled — noisy", COLORS["noisy"]),
    ("fake", "Sampled — fake backend", COLORS["fake"]),
]


def _figure(width: float = 9.0, height: float = 4.5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, alpha=0.6, linewidth=0.6)
    ax.xaxis.label.set_color(FOREGROUND)
    ax.yaxis.label.set_color(FOREGROUND)
    ax.title.set_color(FOREGROUND)
    return fig, ax


def _legend(ax):
    legend = ax.legend(fontsize=8, facecolor=BACKGROUND, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)


def landscape(data: dict, hamiltonian: str, true_ground_energy: float, sweep_label: str):
    """Energy vs one swept parameter, exact against the three samplers."""
    fig, ax = _figure()

    ax.plot(data["x"], data["exact"], color=COLORS["exact"], linewidth=2.5,
            label="Exact (statevector)")
    for key, label, color in [
        ("ideal_sampled", "Sampled — ideal", COLORS["ideal"]),
        ("noisy_sampled", "Sampled — noisy", COLORS["noisy"]),
        ("fake_sampled", "Sampled — fake backend", COLORS["fake"]),
    ]:
        ax.plot(data["x"], data[key], "-o", markersize=3.5, linewidth=1.1,
                color=color, label=label)

    ax.axhline(true_ground_energy, color=COLORS["target"], linestyle="--",
               linewidth=1.4, label=f"ground state {true_ground_energy:.4f}")

    ax.set_xlabel(f"{sweep_label} [rad]")
    ax.set_ylabel(r"$\langle H \rangle$")
    ax.set_title(f"Energy landscape — {hamiltonian}")
    _legend(ax)
    fig.tight_layout()
    return fig


def convergence(result: dict):
    """Energy at each COBYLA iteration against the known ground state."""
    fig, ax = _figure(9.0, 4.0)

    energies = [point["energy"] for point in result["history"]]
    ax.plot(energies, "-o", markersize=3, linewidth=1.1, color=COLORS["exact"],
            label=r"$\langle H \rangle$")
    target = result["true_ground_energy"]
    ax.axhline(target, color=COLORS["target"], linestyle="--", linewidth=1.4,
               label=f"true ground state {target:.4f}")

    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel(r"$\langle H \rangle$")
    ax.set_title(
        f"Convergence — {result['iterations']} iterations, "
        f"final ⟨H⟩ = {result['optimal_energy']:.6f}"
    )
    _legend(ax)
    fig.tight_layout()
    return fig


def histogram(data: dict):
    """Grouped bars of measurement probabilities, with std as error bars."""
    fig, ax = _figure(9.0, 4.8)

    labels = [f"|{bits}⟩" for bits in data["bitstrings"]]
    positions = range(len(labels))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, (key, label, color) in zip(offsets, SERIES):
        heights = data[key] if key == "exact" else data[key]["mean"]
        errors = None if key == "exact" else data[key]["std"]
        ax.bar(
            [p + offset for p in positions], heights, width,
            label=label, color=color, yerr=errors,
            capsize=3 if errors else 0,
            error_kw={"elinewidth": 1.2, "ecolor": MUTED} if errors else {},
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=12, color=FOREGROUND)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Measurement outcomes — mean ± std over "
        f"{data['n_reps']} × {data['n_shots']} shots"
    )
    _legend(ax)
    fig.tight_layout()
    return fig
