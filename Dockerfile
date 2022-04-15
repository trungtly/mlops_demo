# Production-ready MLOps Fraud Detection Service
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlops
RUN chown -R mlops:mlops /app
USER mlops

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV CONFIG_PATH=/app/configs

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Default command to run the API server
CMD ["python", "-m", "uvicorn", "src.fraud_detection.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]