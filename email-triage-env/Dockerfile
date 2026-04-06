FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-submission"
LABEL description="Email Triage OpenEnv — AI agent inbox management environment"
LABEL org.opencontainers.image.title="email-triage-env"
LABEL org.opencontainers.image.version="1.0.0"

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY openenv.yaml .

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server
CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
