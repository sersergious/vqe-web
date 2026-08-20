# One image serving both the API and the frontend.
# Build from the repo root so backend/, frontend/ and vqe/ are all in context:
#   docker build -f backend/Dockerfile -t vqe .

# --- Stage 1: build the frontend to static files -----------------------------
FROM node:22-slim AS frontend
WORKDIR /build

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# --- Stage 2: the Python service ---------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /srv

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY vqe/ vqe/
COPY backend/app/ backend/app/
COPY --from=frontend /build/out/ backend/static/

WORKDIR /srv/backend
EXPOSE 8000

# Single worker: qiskit-aer is CPU-bound and the simulator cache is per-process,
# so extra workers would each rebuild the fake backends (~55 MB and ~0.7s each).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
