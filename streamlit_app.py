"""VQE Explorer — interactive frontend for the experiment scripts in vqe/.

Run with:  streamlit run streamlit_app.py

The scripts in vqe/ are imported unmodified; vqe_app/scenarios.py adapts their
four differing calling conventions to one interface.
"""

from __future__ import annotations

import math
import os
import secrets

import streamlit as st

from vqe_app import charts
from vqe_app import scenarios as sc

st.set_page_config(page_title="VQE Explorer", page_icon="⚛️", layout="wide")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")


def authenticated() -> bool:
    """Single shared password. Unset APP_PASSWORD leaves the app open."""
    if not APP_PASSWORD:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("VQE Explorer")
    password = st.text_input("Password", type="password")
    if password:
        if secrets.compare_digest(password, APP_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not authenticated():
    st.stop()


def widget_key(scenario_id: str, name: str) -> str:
    """Namespaced by scenario so switching scenarios resets to that one's defaults."""
    return f"{scenario_id}::{name}"


# --- Sidebar: pick a scenario and set its parameters --------------------------

scenarios = list(sc.SCENARIOS.values())
labels = {s.id: s.label for s in scenarios}

scenario_id = st.sidebar.selectbox(
    "Scenario",
    options=[s.id for s in scenarios],
    format_func=labels.get,
)
scenario = sc.SCENARIOS[scenario_id]

# Results are per-scenario, so switching scenarios clears the panels.
if st.session_state.get("current_scenario") != scenario_id:
    st.session_state["current_scenario"] = scenario_id
    st.session_state["results"] = {}

results = st.session_state.setdefault("results", {})

# "Load optimal parameters" stashes values here rather than writing the slider
# keys directly: Streamlit forbids assigning a widget's session_state key once
# that widget has been instantiated, so it has to happen before the sliders below.
for _name, _value in st.session_state.pop("pending_params", {}).items():
    st.session_state[widget_key(scenario_id, _name)] = _value

coefficient = None
if scenario.coefficient:
    spec = scenario.coefficient
    coefficient = st.sidebar.slider(
        spec.label,
        min_value=float(spec.min),
        max_value=float(spec.max),
        value=float(spec.default),
        step=0.01,
        key=widget_key(scenario_id, spec.name),
    )

st.sidebar.subheader("Ansatz parameters")
params = [
    st.sidebar.slider(
        spec.label,
        min_value=float(spec.min),
        max_value=float(spec.max),
        value=float(spec.default),
        step=0.01,
        key=widget_key(scenario_id, spec.name),
    )
    for spec in scenario.params
]

st.sidebar.subheader("Sweep")
sweep_index = st.sidebar.selectbox(
    "Parameter to sweep",
    options=range(len(scenario.params)),
    format_func=lambda i: scenario.params[i].label,
    key=widget_key(scenario_id, "sweep_index"),
)
n_points = st.sidebar.slider(
    "Points",
    min_value=4,
    max_value=120,
    value=24,
    step=4,
    help="Each point runs four simulators. 120 takes ~35s on the heaviest scenario.",
    key=widget_key(scenario_id, "n_points"),
)

if not APP_PASSWORD:
    st.sidebar.caption("⚠️ APP_PASSWORD is unset — this instance is open to anyone.")

# --- Header -------------------------------------------------------------------

ground_energy = scenario.ground_energy(coefficient)

st.title("Variational Quantum Eigensolver")
st.caption(
    f"{scenario.hamiltonian} · {scenario.n_qubits} qubit"
    f"{'s' if scenario.n_qubits > 1 else ''} · "
    f"true ground state {ground_energy:.4f}"
)

# --- Actions ------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

if col1.button("Evaluate energy", width="stretch"):
    with st.spinner("Evaluating across all four simulators…"):
        results["energies"] = sc.all_energies(scenario, params, coefficient)

if col2.button("Run VQE", width="stretch"):
    with st.spinner("Optimizing (COBYLA)…"):
        results["vqe"] = sc.run_vqe(scenario, params, coefficient)

if col3.button("Sweep landscape", width="stretch"):
    with st.spinner(f"Sweeping {n_points} points…"):
        results["landscape"] = sc.landscape(
            scenario, sweep_index, params, coefficient, n_points
        )
        results["landscape_label"] = scenario.params[sweep_index].label

if col4.button("Show histogram", width="stretch"):
    with st.spinner("Sampling repeated shot batches…"):
        results["histogram"] = sc.histogram(scenario, params, coefficient)

# --- Results ------------------------------------------------------------------

if energies := results.get("energies"):
    st.subheader("Energy at the current parameters")
    for column, (label, key) in zip(
        st.columns(4),
        [
            ("Exact", "exact"),
            ("Ideal", "ideal_sampled"),
            ("Noisy", "noisy_sampled"),
            ("Fake backend", "fake_sampled"),
        ],
    ):
        column.metric(label, f"{energies[key]:.6f}")

if vqe := results.get("vqe"):
    st.subheader("VQE convergence")
    st.pyplot(charts.convergence(vqe), width="stretch")

    error = abs(vqe["optimal_energy"] - vqe["true_ground_energy"])
    st.caption(
        "optimal parameters: "
        + ", ".join(f"{value:.4f}" for value in vqe["optimal_params"])
        + f"  ·  error vs true ground state: {error:.2e}"
    )
    if st.button("Load optimal parameters into sliders"):
        # COBYLA is unbounded; wrap into the slider's [0, 2π] range.
        st.session_state["pending_params"] = {
            spec.name: float(value % (2 * math.pi))
            for spec, value in zip(scenario.params, vqe["optimal_params"])
        }
        st.rerun()

if landscape := results.get("landscape"):
    st.subheader("Energy landscape")
    st.pyplot(
        charts.landscape(
            landscape,
            scenario.hamiltonian,
            ground_energy,
            results.get("landscape_label", "θ"),
        ),
        width="stretch",
    )

if histogram := results.get("histogram"):
    st.subheader("Measurement outcomes")
    st.pyplot(charts.histogram(histogram), width="stretch")

if not results:
    st.info("Set the parameters in the sidebar, then run one of the actions above.")
