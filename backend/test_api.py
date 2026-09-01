#!/usr/bin/env python3
"""Self-check for the VQE API: one scenario per calling convention.

Run from backend/:  python test_api.py
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# One per family: scalar / pair / pair+coefficient / 2-qubit.
CASES = ["1q_z", "1q_x_plus_z", "1q_b_dot_z", "2q_zz"]


def test_scenarios():
    scenarios = client.get("/api/scenarios").json()
    assert len(scenarios) == 9, f"expected 9 scenarios, got {len(scenarios)}"
    by_id = {s["id"]: s for s in scenarios}
    assert by_id["1q_z"]["coefficient"] is None
    assert by_id["1q_b_dot_z"]["coefficient"]["name"] == "B"
    assert len(by_id["2q_zz"]["params"]) == 8
    return by_id


def test_validation():
    # Wrong parameter count must be rejected, not silently truncated.
    r = client.post(
        "/api/scenarios/2q_zz/evaluate", json={"params": [0.1, 0.2]}
    )
    assert r.status_code == 422, r.status_code
    # Coefficient outside the declared range must be rejected.
    r = client.post(
        "/api/scenarios/1q_b_dot_z/evaluate",
        json={"params": [1.0, 0.5], "coefficient": 99.0},
    )
    assert r.status_code == 422, r.status_code
    r = client.post("/api/scenarios/nope/evaluate", json={"params": [1.0]})
    assert r.status_code == 404, r.status_code


def test_fake_simulators_are_shared():
    """Each fake device must be built once, not once per scenario.

    A FakeSherbrooke-backed AerSimulator costs ~58 MB and Aer does not release
    it, so building one per scenario is the difference between fitting a 512 MB
    instance and being OOM-killed.
    """
    from app import scenarios as sc

    for scenario in sc.SCENARIOS.values():
        sc.simulators(scenario.id)

    distinct = {id(sc.simulators(s.id).fake) for s in sc.SCENARIOS.values()}
    assert len(distinct) == 2, (
        f"expected 2 shared fake simulators (Manila + Sherbrooke), got {len(distinct)}"
    )
    assert len(sc._FAKE_SIMULATORS) == 2, sc._FAKE_SIMULATORS.keys()


def test_readout_error_and_seed():
    """The readout knob must move the noisy series, and only the noisy series.

    The seed guard matters most: handing one seed to all 20 repetitions would
    make them identical and silently collapse every error bar to zero.
    """

    def hist(**extra):
        return client.post(
            "/api/scenarios/1q_z/histogram",
            json={"params": [3.14159265], **extra},
        ).json()

    quiet, loud = hist(readout_error=0.0, seed=3), hist(readout_error=0.25, seed=3)
    assert loud["noisy"]["mean"][0] > quiet["noisy"]["mean"][0] + 0.2, (
        f"readout error did not move the noisy series: "
        f"{quiet['noisy']['mean'][0]} -> {loud['noisy']['mean'][0]}"
    )
    assert quiet["fake"]["mean"] == loud["fake"]["mean"], (
        "readout error leaked into the fake backend, which must stay a device snapshot"
    )

    seeded = hist(readout_error=0.15, seed=7)
    assert seeded == hist(readout_error=0.15, seed=7), "same seed gave a different result"
    assert seeded["readout_error"] == 0.15 and seeded["seed"] == 7, seeded

    # Not "ideal": at theta = pi the noiseless state is exactly |1>, so every
    # shot agrees and its std is legitimately 0, seeded or not.
    for series in ("noisy", "fake"):
        assert all(std > 0 for std in seeded[series]["std"]), (
            f"{series} error bars collapsed: seeding must not make the "
            f"repetitions identical ({seeded[series]['std']})"
        )

    # An omitted seed is still reported, so the run can be repeated afterwards.
    drawn = hist(readout_error=0.05)["seed"]
    assert isinstance(drawn, int) and drawn >= 0, drawn
    # An omitted readout error falls back to the script's own constant.
    assert hist()["readout_error"] == 0.20, hist()["readout_error"]


def check(scenario_id, spec):
    defaults = [p["default"] for p in spec["params"]]
    coefficient = spec["coefficient"]["default"] if spec["coefficient"] else None
    body = {"coefficient": coefficient}

    energies = client.post(
        f"/api/scenarios/{scenario_id}/evaluate",
        json={"params": defaults, **body},
    ).json()
    for key in ("exact", "ideal_sampled", "noisy_sampled", "fake_sampled"):
        assert isinstance(energies[key], float), (scenario_id, key, energies)

    land = client.post(
        f"/api/scenarios/{scenario_id}/landscape",
        json={"sweep_param_index": 0, "fixed_params": defaults, "n_points": 6, **body},
    ).json()
    assert len(land["x"]) == 6 and len(land["exact"]) == 6, land

    result = client.post(
        f"/api/scenarios/{scenario_id}/vqe",
        json={"x0": defaults, **body},
    ).json()
    target = result["true_ground_energy"]
    assert abs(result["optimal_energy"] - target) < 1e-3, (
        f"{scenario_id}: VQE reached {result['optimal_energy']:.6f}, expected {target}"
    )
    assert len(result["optimal_params"]) == len(defaults)
    assert result["iterations"] == len(result["history"]) > 0

    hist = client.post(
        f"/api/scenarios/{scenario_id}/histogram",
        json={"params": defaults, **body},
    ).json()
    n = 2 ** spec["n_qubits"]
    assert len(hist["bitstrings"]) == n, hist["bitstrings"]
    for series in ("exact", "ideal", "noisy", "fake"):
        probs = hist[series] if series == "exact" else hist[series]["mean"]
        assert len(probs) == n
        assert abs(sum(probs) - 1.0) < 1e-6, f"{scenario_id}/{series} sums to {sum(probs)}"

    print(
        f"  {scenario_id:<16} VQE {result['optimal_energy']:+.6f} "
        f"(target {target:+.4f}, {result['iterations']} iters)"
    )


if __name__ == "__main__":
    by_id = test_scenarios()
    test_validation()
    test_fake_simulators_are_shared()
    test_readout_error_and_seed()
    print("metadata + validation + simulator sharing + readout/seed OK\n")
    for scenario_id in CASES:
        check(scenario_id, by_id[scenario_id])
    print("\nAll checks passed.")
