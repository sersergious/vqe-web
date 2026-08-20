#!/usr/bin/env python3
"""Self-check for the VQE API: one scenario per calling convention, plus the auth gate.

Run from backend/:  python test_api.py
"""

import base64
import os

os.environ["AUTH_USERNAME"] = "tester"
os.environ["AUTH_PASSWORD"] = "test-password"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
AUTH = {
    "Authorization": "Basic " + base64.b64encode(b"tester:test-password").decode(),
}

# One per family: scalar / pair / pair+coefficient / 2-qubit.
CASES = ["1q_z", "1q_x_plus_z", "1q_b_dot_z", "2q_zz"]


def wrong(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": "Basic " + token}


def test_auth():
    assert client.get("/api/scenarios").status_code == 401, "anonymous must be refused"
    assert client.get("/api/scenarios", headers=wrong("tester", "nope")).status_code == 401
    assert client.get("/api/scenarios", headers=wrong("nope", "test-password")).status_code == 401
    assert client.get("/api/scenarios", headers={"Authorization": "Basic !!"}).status_code == 401
    assert client.get("/api/scenarios", headers={"Authorization": "Bearer x"}).status_code == 401

    challenge = client.get("/api/scenarios")
    assert "Basic" in challenge.headers.get("www-authenticate", ""), (
        "401 must challenge, so browsers show a login prompt"
    )

    assert client.get("/api/scenarios", headers=AUTH).status_code == 200
    assert client.get("/api/health").status_code == 200, "health stays unauthenticated"


def test_scenarios():
    scenarios = client.get("/api/scenarios", headers=AUTH).json()
    assert len(scenarios) == 9, f"expected 9 scenarios, got {len(scenarios)}"
    by_id = {s["id"]: s for s in scenarios}
    assert by_id["1q_z"]["coefficient"] is None
    assert by_id["1q_b_dot_z"]["coefficient"]["name"] == "B"
    assert len(by_id["2q_zz"]["params"]) == 8
    return by_id


def test_validation():
    # Wrong parameter count must be rejected, not silently truncated.
    r = client.post(
        "/api/scenarios/2q_zz/evaluate", headers=AUTH, json={"params": [0.1, 0.2]}
    )
    assert r.status_code == 422, r.status_code
    # Coefficient outside the declared range must be rejected.
    r = client.post(
        "/api/scenarios/1q_b_dot_z/evaluate",
        headers=AUTH,
        json={"params": [1.0, 0.5], "coefficient": 99.0},
    )
    assert r.status_code == 422, r.status_code
    r = client.post("/api/scenarios/nope/evaluate", headers=AUTH, json={"params": [1.0]})
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


def check(scenario_id, spec):
    defaults = [p["default"] for p in spec["params"]]
    coefficient = spec["coefficient"]["default"] if spec["coefficient"] else None
    body = {"coefficient": coefficient}

    energies = client.post(
        f"/api/scenarios/{scenario_id}/evaluate",
        headers=AUTH,
        json={"params": defaults, **body},
    ).json()
    for key in ("exact", "ideal_sampled", "noisy_sampled", "fake_sampled"):
        assert isinstance(energies[key], float), (scenario_id, key, energies)

    land = client.post(
        f"/api/scenarios/{scenario_id}/landscape",
        headers=AUTH,
        json={"sweep_param_index": 0, "fixed_params": defaults, "n_points": 6, **body},
    ).json()
    assert len(land["x"]) == 6 and len(land["exact"]) == 6, land

    result = client.post(
        f"/api/scenarios/{scenario_id}/vqe",
        headers=AUTH,
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
        headers=AUTH,
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
    test_auth()
    by_id = test_scenarios()
    test_validation()
    test_fake_simulators_are_shared()
    print("auth + metadata + validation + simulator sharing OK\n")
    for scenario_id in CASES:
        check(scenario_id, by_id[scenario_id])
    print("\nAll checks passed.")
