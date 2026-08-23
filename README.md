# VQE Explorer

Interactive Streamlit frontend for the VQE experiment scripts in [`vqe/`](vqe/).

- **`vqe/`** — the original experiment scripts, **unmodified**. They remain runnable
  standalone (`python vqe/vqe_1q_z_noise.py` still writes its PNGs).
- **`vqe_app/scenarios.py`** — adapts the nine scripts' four differing calling
  conventions to one interface, and returns plain dicts.
- **`vqe_app/charts.py`** — matplotlib figures built from those dicts.
- **`streamlit_app.py`** — the page: sidebar widgets, four actions, results.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/streamlit run streamlit_app.py
```

Opens on http://localhost:8501. Set `APP_PASSWORD` to put a password gate in front;
leave it unset and the app runs open, saying so in the sidebar.

## Check it

Covers one scenario per calling convention, the memory invariant, and a smoke test
that drives the real page via Streamlit's `AppTest`:

```bash
.venv/bin/python test_app.py
```

## In a container

```bash
docker compose up --build
```

```bash
APP_PASSWORD=secret docker compose up --build
```

Set `HOST_PORT=8502` if something already holds 8501.

## How it fits together

Streamlit reruns the whole script on every widget interaction, so the four
expensive actions are gated behind buttons and their results kept in
`st.session_state`. Nothing recomputes because you moved an unrelated slider.

Adding a scenario means adding one entry to `SCENARIOS` in
[`vqe_app/scenarios.py`](vqe_app/scenarios.py) — the sidebar builds its sliders
from that entry's parameter specs, so there is no UI change.

## Performance notes

1–2 qubit scenarios, measured inside the container under a hard 512 MB cap:

| | |
|---|---|
| VQE run | < 1s (optimises against the exact statevector) |
| Histogram, heaviest scenario | ~5s |
| Landscape, 24 points | ~7s |
| Landscape, 120 points | ~35s |
| Memory — streamlit + vqe_app imported | 190 MB |
| Memory — all 9 scenarios warm | 255 MB |
| Memory — **peak**, worst case | **351 MB** |
| Memory — live server, typical session | ~219 MB |

"Worst case" is every scenario warm plus the heaviest histogram and a 120-point
sweep. It fits Render's 512 MB starter plan with ~160 MB to spare.

Two things are load-bearing:

- **Fake-backend simulators are shared across scenarios.** One
  `AerSimulator.from_backend(FakeSherbrooke)` costs ~58 MB, and Aer does not
  return that memory when released — so building one per scenario OOM-kills a
  512 MB instance and no eviction policy can recover it. `test_app.py` asserts
  only two ever get built.
- **Actions take tens of seconds.** Streamlit holds a websocket per session, so
  this is fine — but it does rule out serverless hosting.

## Deploying

[`render.yaml`](render.yaml) defines the whole app as one Render service.

1. Push to GitHub.
2. Render dashboard → **New → Blueprint** → pick the repo.
3. It prompts for `APP_PASSWORD` (declared `sync: false`, so it never lands in
   git and the service cannot go live unprotected by accident).

Stay on `starter`, not `free`: free instances spin down when idle, and every
wake-up rebuilds the fake backends. Render health-checks `/_stcore/health`.
