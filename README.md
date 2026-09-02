# VQE Explorer

Interactive frontend for the VQE experiment scripts in [`vqe/`](vqe/). Runs
locally only.

- **`vqe/`** — the experiment scripts. They remain runnable standalone
  (`python vqe/vqe_1q_z_noise.py` still writes its PNGs) and stay the single
  source of truth for their own noise parameters.
- **`backend/`** — FastAPI service that imports those scripts and exposes them
  as JSON.
- **`frontend/`** — Next.js dashboard.

## Run it

Backend (from a venv with `backend/requirements.txt` installed):

```bash
cd backend && ../.venv/bin/uvicorn app.main:app --port 8000
```

Frontend, with hot reload — it proxies `/api/*` to the backend in dev:

```bash
cd frontend && npm install && npm run dev
```

That serves <http://localhost:3000>.

To serve both from one origin instead, build the frontend to static files and
drop them where the backend looks:

```bash
cd frontend && npm run build && cp -r out ../backend/static
```

The backend then serves the dashboard itself at <http://localhost:8000>, and the
dev proxy is no longer involved.

## Check the backend

Covers one scenario per calling convention, request validation, the readout
error and seed controls, and the simulator-sharing invariant below:

```bash
cd backend && ../.venv/bin/python test_api.py
```

## API

| Route | Purpose |
|---|---|
| `GET /api/scenarios` | All 9 scenarios with parameter specs and ground energies |
| `POST /api/scenarios/{id}/evaluate` | ⟨H⟩ at one parameter point, across all four simulators |
| `POST /api/scenarios/{id}/landscape` | Sweep one parameter over [0, 2π] |
| `POST /api/scenarios/{id}/vqe` | Run COBYLA, returns optimal params + iteration history |
| `POST /api/scenarios/{id}/histogram` | Measurement probabilities, mean ± std over repeated shot batches |

The three sampling routes (`evaluate`, `landscape`, `histogram`) also accept
`readout_error` — the symmetric measurement bit-flip probability, `0.00`–`0.25`,
defaulting to the scenario script's own constant — and `seed`, defaulting to a
value the server draws. Both are echoed in the response, so any figure names the
settings that produced it and can be regenerated exactly. `vqe` accepts neither:
it optimizes the exact expectation and never draws shots.

Adding a scenario means adding one entry to `SCENARIOS` in
[`backend/app/scenarios.py`](backend/app/scenarios.py); the frontend renders its
sliders from the returned parameter specs with no UI changes.

## Performance notes

Measured on 1–2 qubit scenarios:

| | |
|---|---|
| Memory, idle | ~307 MB |
| Memory, all 9 scenarios exercised | ~334 MB (stable) |
| VQE run | < 1s (optimises against the exact statevector) |
| Histogram, heaviest scenario | ~5s |
| Landscape, 24 points | ~7s |
| Landscape, 120 points | ~35s |

Two things follow from this, and both are load-bearing:

- **Fake-backend simulators are shared across scenarios.** One
  `AerSimulator.from_backend(FakeSherbrooke)` costs ~58 MB, and Aer does not
  return that memory when released — so building one per scenario would add a
  few hundred MB that nothing can reclaim. `test_api.py` asserts only two ever
  get built.
- **The noisy simulator is not cached.** It depends on the request's readout
  error, and rebuilding it costs 0.4 ms (1q) / 3.0 ms (2q) against ~300 ms per
  simulated point — so a cache keyed on the readout error would save ~1% and
  grow without bound.
