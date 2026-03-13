# 🚀 ML Anomaly Detection System - Production Docker Image
FROM python:3.11-slim

# Metadata
LABEL maintainer="ML Team"
LABEL version="5.0.0"
LABEL description="Enterprise-grade anomaly detection for crypto trading"

# Environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY package.json ./

# Copy configuration and docs
COPY README.md ./
COPY examples/ ./examples/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Default command - start API server
CMD ["python", "-m", "uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]