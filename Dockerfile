# Multi-stage Dockerfile for sinXdetect Application
# This Dockerfile builds both frontend and backend for deployment to Digital Ocean

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy frontend source code
COPY frontend/ .

# Build argument for API URL (can be overridden at build time)
ARG VITE_API_URL=https://api.sinxdetect.movindu.com
ENV VITE_API_URL=${VITE_API_URL}

# Build the frontend application
RUN npm run build

# ============================================
# Stage 2: Backend with Frontend served by nginx
# ============================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including nginx
RUN apt-get update && apt-get install -y \
    build-essential \
    nginx \
    supervisor \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for better caching
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
# Install TensorFlow first, then transformers to ensure TF models are available
RUN pip install --no-cache-dir "tensorflow>=2.15.0,<2.16.0" "tf-keras>=2.15.0,<2.16.0" && \
    pip install --no-cache-dir "transformers[tensorflow]>=4.35.0,<5.0.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Create directory for ML models (will be mounted or copied)
RUN mkdir -p /app/ml/models

# Copy ML models if they exist (for deployments that include models)
COPY ml/models/ ./ml/models/

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Remove default nginx configuration if it exists
RUN rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/sinxdetect.conf

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/sinxdetect.conf

# Expose port 80 (nginx serves both frontend and proxies to backend)
EXPOSE 80

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/ml/models/sinbert_sinhala_classifier
ENV UVICORN_WORKERS=1

# Health check - longer start period for ML model loading
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

# Start supervisor which manages both nginx and backend
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
