# Deploying

The image has been built and run locally, but nothing here has been deployed to a
real Render account or VPS. These are the steps; run them yourself, since they
need your credentials.

## Render (recommended)

[`render.yaml`](../render.yaml) defines the whole thing as one service.

1. Push this repo to GitHub.
2. Render dashboard → **New → Blueprint** → pick the repo. It reads `render.yaml`.
3. It will prompt for `AUTH_USERNAME` and `AUTH_PASSWORD` — they are declared
   `sync: false`, so they never live in git and the service cannot go live
   unprotected by accident. Generate a password with:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(24))"
```

4. First build takes a while (it compiles the frontend, then installs qiskit).
   Render health-checks `/api/health`, which is deliberately public.

**Stay on the `starter` plan, not `free`.** Free instances spin down when idle,
and every wake-up re-pays the image start plus rebuilding the fake backends.
Starter's 512 MB is enough — measured ~334 MB steady-state — but only because
the fake simulators are shared; see the performance notes in the root README.

If you later push to many more qubits, memory becomes the binding constraint and
a VPS gives far more RAM per pound than the next plan up. That is the point to
switch, and the compose setup below is already there for it.

## Self-hosting instead

Same image, on any box with Docker and a domain pointed at it:

```bash
cp backend/.env.example backend/.env   # then fill in AUTH_* and DOMAIN
```

```bash
docker compose --env-file backend/.env -f backend/docker-compose.yml up -d --build
```

Caddy obtains and renews TLS automatically, provided the domain's A record points
at the host and ports 80/443 are open. Verify with:

```bash
curl https://your-domain.example/api/health
```

## Rotating credentials

Change `AUTH_USERNAME`/`AUTH_PASSWORD` and restart. Nothing else caches them —
the browser will simply prompt again.
