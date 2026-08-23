# Build from the repo root:  docker build -t vqe-explorer .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vqe/ vqe/
COPY vqe_app/ vqe_app/
COPY .streamlit/ .streamlit/
COPY streamlit_app.py .

EXPOSE 8501

HEALTHCHECK --start-period=40s --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Streamlit holds a websocket per session, so long computations stream progress
# instead of racing an HTTP timeout. PORT is set by Render; 8501 is the default.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
