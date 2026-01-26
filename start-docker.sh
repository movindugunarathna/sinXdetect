#!/bin/bash

# Docker startup script for sinXdetect application

set -e

echo "🚀 Starting sinXdetect Application with Docker..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Parse command line arguments
MODE=${1:-production}

if [ "$MODE" = "dev" ] || [ "$MODE" = "development" ]; then
    echo "📦 Building and starting in DEVELOPMENT mode..."
    echo "   - Backend with hot-reload on http://localhost:8000"
    echo "   - Frontend with hot-reload on http://localhost:5173"
    echo ""
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
else
    echo "📦 Building and starting in PRODUCTION mode..."
    echo "   - Backend API on http://localhost:8000"
    echo "   - Frontend on http://localhost:3000"
    echo ""
    docker compose up --build -d
    
    echo ""
    echo "✅ Services started successfully!"
    echo ""
    echo "📊 View logs with: docker compose logs -f"
    echo "🛑 Stop services with: docker compose down"
    echo ""
    echo "🌐 Open the application at: http://localhost:3000"
fi
