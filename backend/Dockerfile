# Backend Dockerfile for Flask/FastAPI Application
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install TensorFlow and transformers with pinned versions for compatibility
RUN pip install --no-cache-dir \
    "tensorflow>=2.15.0,<2.16.0" \
    "tf-keras>=2.15.0,<2.16.0" \
    "transformers>=4.36.0,<4.37.0" \
    "tokenizers>=0.15.0,<0.16.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Create directory for ML models (will be mounted from host or copied)
RUN mkdir -p /app/ml/models

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/ml/models/sinbert_sinhala_classifier

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application with uvicorn
# For production: remove --reload flag
# For development: keep --reload flag
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
