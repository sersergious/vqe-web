#!/usr/bin/env python3
"""Self-check for the VQE app.

Covers one scenario per calling convention, the memory invariant that keeps the
app inside a 512 MB instance, and a smoke test that the Streamlit page runs.

Run:  python test_app.py
"""

import os

from vqe_app import scenarios as sc

# One per family: scalar / pair / pair+coefficient / 2-qubit.
CASES = ["1q_z", "1q_x_plus_z", "1q_b_dot_z", "2q_zz"]


def test_registry():
    assert len(sc.SCENARIOS) == 9, f"expected 9 scenarios, got {len(sc.SCENARIOS)}"
    assert sc.SCENARIOS["1q_z"].coefficient is None
    assert sc.SCENARIOS["1q_b_dot_z"].coefficient.name == "B"
    assert len(sc.SCENARIOS["2q_zz"].params) == 8
    # Ground energy of -B*Z tracks the coefficient rather than being constant.
    assert sc.SCENARIOS["1q_b_dot_z"].ground_energy(-2.5) == -2.5
    assert sc.SCENARIOS["1q_b_dot_z"].ground_energy(1.5) == -1.5


def test_fake_simulators_are_shared():
    """Each fake device must be built once, not once per scenario.

    A FakeSherbrooke-backed AerSimulator costs ~58 MB and Aer does not release
    it, so building one per scenario is the difference between fitting a 512 MB
    instance and being OOM-killed.
    """
    for scenario in sc.SCENARIOS.values():
        sc.simulators(scenario.id)

    distinct = {id(sc.simulators(s.id).fake) for s in sc.SCENARIOS.values()}
    assert len(distinct) == 2, (
        f"expected 2 shared fake simulators (Manila + Sherbrooke), got {len(distinct)}"
    )


def check(scenario_id):
    scenario = sc.SCENARIOS[scenario_id]
    params = [spec.default for spec in scenario.params]
    coefficient = scenario.coefficient.default if scenario.coefficient else None

    energies = sc.all_energies(scenario, params, coefficient)
    for key in ("exact", "ideal_sampled", "noisy_sampled", "fake_sampled"):
        assert isinstance(energies[key], float), (scenario_id, key, energies)

    land = sc.landscape(scenario, 0, params, coefficient, n_points=6)
    assert len(land["x"]) == 6 and len(land["exact"]) == 6, land

    result = sc.run_vqe(scenario, params, coefficient)
    target = result["true_ground_energy"]
    assert abs(result["optimal_energy"] - target) < 1e-3, (
        f"{scenario_id}: VQE reached {result['optimal_energy']:.6f}, expected {target}"
    )
    assert result["iterations"] == len(result["history"]) > 0
    assert len(result["optimal_params"]) == len(params)

    hist = sc.histogram(scenario, params, coefficient)
    n = 2 ** scenario.n_qubits
    assert len(hist["bitstrings"]) == n, hist["bitstrings"]
    for series in ("exact", "ideal", "noisy", "fake"):
        probs = hist[series] if series == "exact" else hist[series]["mean"]
        assert len(probs) == n
        assert abs(sum(probs) - 1.0) < 1e-6, f"{scenario_id}/{series} sums to {sum(probs)}"

    # The figures must build from these dicts without touching the filesystem.
    from vqe_app import charts

    charts.convergence(result)
    charts.landscape(land, scenario.hamiltonian, target, scenario.params[0].label)
    charts.histogram(hist)

    print(
        f"  {scenario_id:<16} VQE {result['optimal_energy']:+.6f} "
        f"(target {target:+.4f}, {result['iterations']} iters)"
    )


def test_app_renders():
    """Smoke test: the Streamlit page runs top to bottom without raising."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("streamlit_app.py", default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert len(app.sidebar.selectbox[0].options) == 9
    assert len(app.button) >= 4, "expected the four action buttons"

    # Driving a real action exercises the widget -> scenarios -> chart path.
    def button(app, label):
        found = [b for b in app.button if b.label == label]
        assert found, f"no button labelled {label!r}"
        return found[0]

    button(app, "Run VQE").click().run()
    assert not app.exception, app.exception


def test_app_switches_scenarios():
    """Switching to an 8-parameter scenario must re-render its sliders and run."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("streamlit_app.py", default_timeout=180)
    app.run()
    assert len(app.sidebar.slider) == 2, "1q_z: one angle + the sweep Points slider"

    app.sidebar.selectbox[0].select("2q_xx_yy_zz").run()
    assert not app.exception, app.exception
    # 8 ansatz parameters + the sweep "Points" slider.
    assert len(app.sidebar.slider) == 9, [s.label for s in app.sidebar.slider]

    [b for b in app.button if b.label == "Run VQE"][0].click().run()
    assert not app.exception, app.exception

    # Loading unbounded COBYLA output back into [0, 2pi] sliders must not raise.
    load = [b for b in app.button if b.label == "Load optimal parameters into sliders"]
    assert load, "expected the load-optimal button after a VQE run"
    load[0].click().run()
    assert not app.exception, app.exception


def test_password_gate():
    """With APP_PASSWORD set, nothing renders until the right password is given."""
    from streamlit.testing.v1 import AppTest

    os.environ["APP_PASSWORD"] = "s3cret"
    try:
        app = AppTest.from_file("streamlit_app.py", default_timeout=180)
        app.run()
        assert not app.exception, app.exception
        assert app.text_input, "expected a password prompt"
        assert not app.sidebar.selectbox, "app must stay hidden until unlocked"

        app.text_input[0].input("wrong").run()
        assert app.error, "a wrong password must show an error"
        assert not app.sidebar.selectbox, "app must stay hidden after a bad password"

        app.text_input[0].input("s3cret").run()
        assert not app.exception, app.exception
        assert app.sidebar.selectbox, "correct password must reveal the app"
    finally:
        os.environ.pop("APP_PASSWORD", None)


if __name__ == "__main__":
    test_registry()
    test_fake_simulators_are_shared()
    print("registry + simulator sharing OK\n")
    for scenario_id in CASES:
        check(scenario_id)
    print("\nrendering the Streamlit page…")
    test_app_renders()
    test_app_switches_scenarios()
    test_password_gate()
    print("app renders, switches scenarios, loads optimal params, gates on password")
    print("\nAll checks passed.")
