# Docker Setup for sinXdetect

This document explains how to run the sinXdetect application using Docker containers.

## Architecture

The application consists of three main components:

1. **Backend**: FastAPI application serving the ML classification API (Port 8000)
2. **Frontend**: React + Vite application served by Nginx (Port 3000)
3. **ML Models**: Pre-trained models stored in the `ml/models` directory

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose 1.29+ installed
- At least 4GB of free disk space
- ML models present in `ml/models/` directory

## Quick Start

### Production Mode

Run the entire application stack:

```bash
docker-compose up -d
```

This will:

- Build and start the backend service on `http://localhost:8000`
- Build and start the frontend service on `http://localhost:3000`
- Mount ML models as read-only volumes

Access the application at: **http://localhost:3000**

### Development Mode

For development with hot-reload:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This will:

- Enable hot-reload for backend (FastAPI auto-reload)
- Enable hot-reload for frontend (Vite HMR)
- Frontend dev server on `http://localhost:5173`
- Backend API on `http://localhost:8000`

## Available Commands

### Build Services

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Start Services

```bash
# Start in foreground
docker-compose up

# Start in background (detached)
docker-compose up -d

# Start specific service
docker-compose up backend
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### View Logs

```bash
# View all logs
docker-compose logs

# Follow logs
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs frontend
```

### Execute Commands in Containers

```bash
# Access backend shell
docker-compose exec backend bash

# Access frontend shell
docker-compose exec frontend sh

# Run Python commands in backend
docker-compose exec backend python -c "print('Hello')"
```

## Configuration

### Backend Environment Variables

Edit `docker-compose.yml` to configure:

- `MODEL_PATH`: Path to the ML model directory
- `PYTHONUNBUFFERED`: Set to 1 for immediate stdout/stderr

### Frontend Environment Variables

To change the API URL, modify the build args in `docker-compose.yml`:

```yaml
args:
  - VITE_API_URL=http://your-api-url:8000
```

Or create a `.env` file in the frontend directory:

```
VITE_API_URL=http://localhost:8000
```

## Volumes

The following volumes are mounted:

- `./ml/models:/app/ml/models:ro` - ML models (read-only)
- `./backend:/app` - Backend source code (dev mode only)
- `./frontend:/app` - Frontend source code (dev mode only)

## Networking

Services communicate through a dedicated Docker network (`sinxdetect-network`):

- Backend can be accessed at `http://backend:8000` from within the network
- Frontend can be accessed at `http://frontend:80` from within the network

## Health Checks

Both services include health checks:

- Backend: Checks `/health` endpoint every 30 seconds
- Frontend: Checks nginx server every 30 seconds

View health status:

```bash
docker-compose ps
```

## Troubleshooting

### Backend won't start

- Check if ML models exist in `ml/models/` directory
- Check backend logs: `docker-compose logs backend`
- Verify Python dependencies in `requirements.txt`

### Frontend won't build

- Check if `package.json` exists
- Clear node_modules: `docker-compose down -v && docker-compose build --no-cache frontend`
- Check frontend logs: `docker-compose logs frontend`

### Port conflicts

If ports 8000 or 3000 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
ports:
  - '8001:8000' # Change host port to 8001
```

### Model loading issues

Ensure models are properly structured:

```
ml/models/
└── sinbert_sinhala_classifier/
    ├── config.json
    ├── tf_model.h5
    ├── tokenizer_config.json
    └── ...
```

## Production Deployment

For production deployment:

1. Remove development volume mounts from `docker-compose.yml`
2. Set proper CORS origins in backend `app.py`
3. Use environment-specific `.env` files
4. Enable HTTPS with a reverse proxy (nginx/traefik)
5. Consider using Docker Swarm or Kubernetes for orchestration

### Example Production docker-compose.yml

```yaml
services:
  backend:
    image: sinxdetect-backend:latest
    volumes:
      - ./ml/models:/app/ml/models:ro
    environment:
      - MODEL_PATH=/app/ml/models/sinbert_sinhala_classifier
    restart: always

  frontend:
    image: sinxdetect-frontend:latest
    restart: always
```

## Resource Requirements

Recommended resources:

- **Memory**: 2GB minimum (4GB recommended)
- **CPU**: 2 cores minimum
- **Disk**: 4GB for images + models

## Security Considerations

1. Change CORS settings in production
2. Use secrets for sensitive configuration
3. Run containers as non-root user
4. Keep base images updated
5. Scan images for vulnerabilities: `docker scan sinxdetect-backend`

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/)
- [Vite Docker Guide](https://vitejs.dev/guide/static-deploy.html)
