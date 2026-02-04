#!/bin/bash

# Docker stop script for sinXdetect application

set -e

echo "🛑 Stopping sinXdetect Application..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running."
    exit 1
fi

# Parse command line arguments
MODE=${1:-all}
CLEAN=${2:-no}

# Function to stop services
stop_services() {
    local compose_file=$1
    local clean=$2
    
    if [ "$clean" = "clean" ] || [ "$clean" = "--clean" ]; then
        docker compose -f "$compose_file" down -v --remove-orphans 2>/dev/null || true
    else
        docker compose -f "$compose_file" down --remove-orphans 2>/dev/null || true
    fi
}

if [ "$MODE" = "clean" ] || [ "$MODE" = "--clean" ]; then
    echo "🧹 Stopping ALL services and removing volumes..."
    stop_services "docker-compose.yml" "clean"
    stop_services "docker-compose.dev.yml" "clean"
    stop_services "docker-compose.prod.yml" "clean"
    echo ""
    echo "✅ All services stopped and volumes removed!"
elif [ "$MODE" = "prod" ] || [ "$MODE" = "production" ]; then
    echo "🛑 Stopping PRODUCTION services..."
    stop_services "docker-compose.prod.yml" "$CLEAN"
    echo ""
    echo "✅ Production services stopped!"
elif [ "$MODE" = "dev" ] || [ "$MODE" = "development" ]; then
    echo "🛑 Stopping DEVELOPMENT services..."
    stop_services "docker-compose.dev.yml" "$CLEAN"
    echo ""
    echo "✅ Development services stopped!"
elif [ "$MODE" = "local" ]; then
    echo "🛑 Stopping LOCAL services..."
    stop_services "docker-compose.yml" "$CLEAN"
    echo ""
    echo "✅ Local services stopped!"
else
    echo "🛑 Stopping ALL services..."
    stop_services "docker-compose.yml" "no"
    stop_services "docker-compose.dev.yml" "no"
    stop_services "docker-compose.prod.yml" "no"
    echo ""
    echo "✅ All services stopped!"
    echo ""
    echo "💡 To remove volumes as well, run: ./stop-docker.sh clean"
fi

echo ""
echo "🔄 To restart, run: ./start-docker.sh [dev|local|prod]"
