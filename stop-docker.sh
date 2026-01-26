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
REMOVE_VOLUMES=${1:-no}

if [ "$REMOVE_VOLUMES" = "clean" ] || [ "$REMOVE_VOLUMES" = "--clean" ]; then
    echo "🧹 Stopping services and removing volumes..."
    docker compose down -v
    echo ""
    echo "✅ Services stopped and volumes removed!"
else
    echo "🛑 Stopping services..."
    docker compose down
    echo ""
    echo "✅ Services stopped!"
    echo ""
    echo "💡 To remove volumes as well, run: ./stop-docker.sh clean"
fi

echo ""
echo "🔄 To restart, run: ./start-docker.sh"
