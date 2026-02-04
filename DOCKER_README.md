# Docker Setup for sinXdetect

This document explains how to run the sinXdetect application using Docker containers.

## Architecture

The application consists of three main components:

1. **Backend**: FastAPI application serving the ML classification API (Port 8000)
2. **Frontend**: React + Vite application served by Nginx (Port 3000/80)
3. **ML Models**: Pre-trained models stored in the `ml/models` directory

## Environment Overview

| Environment     | Compose File              | Frontend URL                   | Backend API URL                    |
| --------------- | ------------------------- | ------------------------------ | ---------------------------------- |
| **Development** | `docker-compose.dev.yml`  | http://localhost:5173          | http://localhost:8000              |
| **Local**       | `docker-compose.yml`      | http://localhost:3000          | http://localhost:3000/api          |
| **Production**  | `docker-compose.prod.yml` | https://sinxdetect.movindu.com | https://api.sinxdetect.movindu.com |

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose 1.29+ installed
- At least 4GB of free disk space
- ML models present in `ml/models/` directory

## Quick Start

### Using Start Scripts (Recommended)

```bash
# Development mode (with hot-reload)
./start-docker.sh dev

# Local testing (combined container on port 3000)
./start-docker.sh local

# Production deployment
./start-docker.sh prod
```

On Windows:

```cmd
start-docker.bat dev
start-docker.bat local
start-docker.bat prod
```

### Development Mode

For development with hot-reload:

```bash
docker compose -f docker-compose.dev.yml up --build --remove-orphans
```

This will:

- Enable hot-reload for backend (FastAPI auto-reload)
- Enable hot-reload for frontend (Vite HMR)
- Frontend dev server on `http://localhost:5173`
- Backend API on `http://localhost:8000`

### Local Testing Mode

Test the combined production-like container locally:

```bash
docker compose up --build -d --remove-orphans
```

This will:

- Build the combined frontend + backend container
- Application available at `http://localhost:3000`
- API accessible at `http://localhost:3000/api/`

### Production Mode

Deploy to production:

```bash
docker compose -f docker-compose.prod.yml up --build -d --remove-orphans
```

This will:

- Build optimized production container
- Frontend URL: `https://sinxdetect.movindu.com`
- Backend API: `https://api.sinxdetect.movindu.com`
- Container exposes port 80

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
./stop-docker.sh

# Stop specific environment
./stop-docker.sh dev
./stop-docker.sh local
./stop-docker.sh prod

# Stop and remove volumes
./stop-docker.sh clean
```

On Windows:

```cmd
stop-docker.bat
stop-docker.bat dev
stop-docker.bat prod
stop-docker.bat clean
```

Or using docker compose directly:

```bash
# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v
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

### Environment-specific API URLs

The frontend connects to different API URLs based on the environment:

| Environment | VITE_API_URL                         |
| ----------- | ------------------------------------ |
| Development | `http://localhost:8000`              |
| Local       | `http://localhost:3000/api`          |
| Production  | `https://api.sinxdetect.movindu.com` |

### Backend Environment Variables

Edit the appropriate `docker-compose*.yml` to configure:

- `MODEL_PATH`: Path to the ML model directory
- `PYTHONUNBUFFERED`: Set to 1 for immediate stdout/stderr
- `UVICORN_WORKERS`: Number of worker processes (1 for dev, 2+ for prod)

### Frontend Environment Variables

The API URL is set at build time via the `VITE_API_URL` build argument:

**Development** (`docker-compose.dev.yml`):

```yaml
environment:
  - VITE_API_URL=http://localhost:8000
```

**Local** (`docker-compose.yml`):

```yaml
args:
  - VITE_API_URL=http://localhost:3000/api
```

**Production** (`docker-compose.prod.yml`):

```yaml
args:
  - VITE_API_URL=https://api.sinxdetect.movindu.com
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

For production deployment on Digital Ocean or similar:

### Production URLs

- **Frontend**: https://sinxdetect.movindu.com
- **Backend API**: https://api.sinxdetect.movindu.com

### Deployment Steps

1. Deploy using the production compose file:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

2. Configure your reverse proxy/load balancer to:

   - Route `sinxdetect.movindu.com` to port 80 (frontend)
   - Route `api.sinxdetect.movindu.com` to port 80 (backend via nginx proxy)

3. Set up SSL certificates (e.g., using Certbot/Let's Encrypt)

### Docker Compose Files Summary

| File                      | Purpose            | Use Case                           |
| ------------------------- | ------------------ | ---------------------------------- |
| `docker-compose.yml`      | Base/Local testing | Testing combined container locally |
| `docker-compose.dev.yml`  | Development        | Hot-reload, separate containers    |
| `docker-compose.prod.yml` | Production         | Optimized for deployment           |

### Example Production Deployment

```bash
# On your production server
git pull origin main
./start-docker.sh prod

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Stop production
./stop-docker.sh prod
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
