@echo off
REM Docker startup script for sinXdetect application on Windows
REM Uses the unified multi-stage Dockerfile

echo.
echo Starting sinXdetect Application with Docker...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Parse command line arguments
set MODE=%1
if "%MODE%"=="" set MODE=production

if /i "%MODE%"=="dev" goto development
if /i "%MODE%"=="development" goto development
goto production

:development
echo Building and starting in DEVELOPMENT mode...
echo    - Backend with hot-reload on http://localhost:8000
echo    - Frontend with hot-reload on http://localhost:5173
echo.
docker compose -f docker-compose.dev.yml up --build
goto end

:production
echo Building and starting in PRODUCTION mode...
echo    - Using unified Dockerfile (frontend + backend combined)
echo    - Application available on http://localhost:3000
echo.
docker compose up --build -d

echo.
echo Services started successfully!
echo.
echo View logs with: docker compose logs -f
echo Stop services with: stop-docker.bat
echo.
echo Open the application at: http://localhost:3000
echo.
echo Note: First startup may take 2-3 minutes for ML model loading.
echo.
goto end

:end
