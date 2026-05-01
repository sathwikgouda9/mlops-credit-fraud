# Dockerfile
# Multi-stage build: keeps the final image lean by separating
# the build environment from the runtime environment.

# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for some ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Credit Fraud Detector API"
LABEL org.opencontainers.image.description="FastAPI + RandomForest MLOps demo"

WORKDIR /app

# Copy installed packages from builder (keeps image ~300MB instead of ~600MB)
COPY --from=builder /install /usr/local

# Copy application code
COPY api/        ./api/
COPY src/        ./src/
COPY params.yaml .

# Copy the trained model artifact
# (In production, mount this as a volume or pull from model registry)
COPY models/     ./models/

# Copy the fitted scaler (needed at inference time)
COPY data/processed/scaler.pkl ./data/processed/scaler.pkl

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check — Docker / K8s uses this to know the container is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
