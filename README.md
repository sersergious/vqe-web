# VQE Explorer

Interactive frontend for the VQE experiment scripts in [`vqe/`](vqe/).

- **`vqe/`** — the original experiment scripts, **unmodified**. They remain runnable
  standalone (`python vqe/vqe_1q_z_noise.py` still writes its PNGs).
- **`backend/`** — FastAPI service that imports those scripts, exposes them as JSON,
  and serves the built frontend.
- **`frontend/`** — Next.js dashboard, built to static files.

One container serves everything on one origin, so there is no CORS and no proxy
hop. The app is unauthenticated — everything it exposes is public to anyone who
can reach it, which is fine for local use but not for an open deployment.

## Run it locally

Backend (from a venv with `backend/requirements.txt` installed):

```bash
cd backend && ../.venv/bin/uvicorn app.main:app --port 8000
```

Frontend, with hot reload — it proxies `/api/*` to the backend in dev:

```bash
cd frontend && npm install && npm run dev
```

## Check the backend

Covers one scenario per calling convention, request validation, and the
simulator-sharing invariant that keeps memory in budget:

```bash
cd backend && ../.venv/bin/python test_api.py
```

## Run the real thing

[`docker-compose.yml`](docker-compose.yml) runs the production image exactly as
Render will, including a 512 MB cap so a memory regression fails here rather than
in production:

```bash
docker compose up --build
```

That serves http://localhost:8000. Set `HOST_PORT=8001` if something already
holds port 8000. Check the memory budget while clicking around with:

```bash
docker stats vqe-local --no-stream --format '{{.MemUsage}}'
```

## API

| Route | Purpose |
|---|---|
| `GET /api/scenarios` | All 9 scenarios with parameter specs and ground energies |
| `POST /api/scenarios/{id}/evaluate` | ⟨H⟩ at one parameter point, across all four simulators |
| `POST /api/scenarios/{id}/landscape` | Sweep one parameter over [0, 2π] |
| `POST /api/scenarios/{id}/vqe` | Run COBYLA, returns optimal params + iteration history |
| `POST /api/scenarios/{id}/histogram` | Measurement probabilities, mean ± std over repeated shot batches |

Adding a scenario means adding one entry to `SCENARIOS` in
[`backend/app/scenarios.py`](backend/app/scenarios.py); the frontend renders its
sliders from the returned parameter specs with no UI changes.

## Performance notes

Measured in the container, 1–2 qubit scenarios:

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
  return that memory when released — so building one per scenario OOM-kills a
  512 MB instance and no eviction policy can save it. `test_api.py` asserts only
  two ever get built.
- **Requests are synchronous and can take ~35s.** That rules out serverless
  hosting; this needs a long-lived process.

## Deploying

See [`backend/README.md`](backend/README.md).
