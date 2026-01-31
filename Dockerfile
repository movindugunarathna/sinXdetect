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
ARG VITE_API_URL=/api
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
# Note: If ml/models/ doesn't exist in your build context, you can either:
# 1. Create an empty ml/models/ directory before building
# 2. Mount models at runtime using volumes
# 3. Remove this COPY if models will be provided externally
COPY ml/models/ ./ml/models/

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

# Remove default nginx configuration if it exists
RUN if [ -f /etc/nginx/sites-enabled/default ]; then rm /etc/nginx/sites-enabled/default; fi

# Create nginx configuration for serving frontend and proxying API
RUN cat > /etc/nginx/conf.d/sinxdetect.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript application/json;

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Handle client-side routing for SPA
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
EOF

# Create supervisor configuration to run both nginx and uvicorn
# UVICORN_WORKERS can be overridden at runtime via environment variable
RUN cat > /etc/supervisor/conf.d/sinxdetect.conf << 'EOF'
[supervisord]
nodaemon=true
user=root

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stderr_logfile=/var/log/nginx/error.log
stdout_logfile=/var/log/nginx/access.log

[program:backend]
command=/bin/sh -c "uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/backend.err.log
stdout_logfile=/var/log/backend.out.log
environment=PYTHONUNBUFFERED="1",MODEL_PATH="/app/ml/models/sinbert_sinhala_classifier"
EOF

# Expose port 80 (nginx serves both frontend and proxies to backend)
EXPOSE 80

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/ml/models/sinbert_sinhala_classifier
ENV UVICORN_WORKERS=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

# Start supervisor which manages both nginx and backend
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
