#!/bin/bash

# Docker startup script for sinXdetect application
# Uses the unified multi-stage Dockerfile

set -e

echo "🚀 Starting sinXdetect Application with Docker..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Parse command line arguments
MODE=${1:-dev}

if [ "$MODE" = "dev" ] || [ "$MODE" = "development" ]; then
    echo "📦 Building and starting in DEVELOPMENT mode (hot-reload)..."
    echo "   - Backend with hot-reload on http://localhost:8000"
    echo "   - Frontend with hot-reload on http://localhost:5173"
    echo ""
    docker compose -f docker-compose.dev.yml up --build --remove-orphans
elif [ "$MODE" = "local" ]; then
    echo "📦 Building and starting in LOCAL mode (combined container)..."
    echo "   - Application available on http://localhost:3000"
    echo "   - API available on http://localhost:3000/api/"
    echo ""
    docker compose up --build -d --remove-orphans
    
    echo ""
    echo "✅ Services started successfully!"
    echo ""
    echo "📊 View logs with: docker compose logs -f"
    echo "🛑 Stop services with: ./stop-docker.sh"
    echo ""
    echo "🌐 Open the application at: http://localhost:3000"
    echo ""
    echo "⏳ Note: First startup may take 2-3 minutes for ML model loading."
elif [ "$MODE" = "prod" ] || [ "$MODE" = "production" ]; then
    echo "📦 Building and starting in PRODUCTION mode..."
    echo "   - Frontend URL: https://sinxdetect.movindu.com"
    echo "   - Backend API:  https://api.sinxdetect.movindu.com"
    echo ""
    docker compose -f docker-compose.prod.yml up --build -d --remove-orphans
    
    echo ""
    echo "✅ Production services started successfully!"
    echo ""
    echo "📊 View logs with: docker compose -f docker-compose.prod.yml logs -f"
    echo "🛑 Stop services with: ./stop-docker.sh prod"
    echo ""
    echo "🌐 Application URLs:"
    echo "   - Frontend: https://sinxdetect.movindu.com"
    echo "   - API:      https://api.sinxdetect.movindu.com"
    echo ""
    echo "⏳ Note: First startup may take 2-3 minutes for ML model loading."
else
    echo "❌ Unknown mode: $MODE"
    echo ""
    echo "Usage: ./start-docker.sh [mode]"
    echo ""
    echo "Modes:"
    echo "  dev, development  - Start with hot-reload (frontend:5173, backend:8000)"
    echo "  local             - Start combined container locally (port 3000)"
    echo "  prod, production  - Start production build (port 80)"
    echo ""
    exit 1
fi
