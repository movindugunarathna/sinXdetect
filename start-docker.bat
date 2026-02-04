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
if "%MODE%"=="" set MODE=dev

if /i "%MODE%"=="dev" goto development
if /i "%MODE%"=="development" goto development
if /i "%MODE%"=="local" goto local
if /i "%MODE%"=="prod" goto production
if /i "%MODE%"=="production" goto production
goto usage

:development
echo Building and starting in DEVELOPMENT mode (hot-reload)...
echo    - Backend with hot-reload on http://localhost:8000
echo    - Frontend with hot-reload on http://localhost:5173
echo.
docker compose -f docker-compose.dev.yml up --build --remove-orphans
goto end

:local
echo Building and starting in LOCAL mode (combined container)...
echo    - Application available on http://localhost:3000
echo    - API available on http://localhost:3000/api/
echo.
docker compose up --build -d --remove-orphans

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

:production
echo Building and starting in PRODUCTION mode...
echo    - Frontend URL: https://sinxdetect.movindu.com
echo    - Backend API:  https://api.sinxdetect.movindu.com
echo.
docker compose -f docker-compose.prod.yml up --build -d --remove-orphans

echo.
echo Production services started successfully!
echo.
echo View logs with: docker compose -f docker-compose.prod.yml logs -f
echo Stop services with: stop-docker.bat prod
echo.
echo Application URLs:
echo    - Frontend: https://sinxdetect.movindu.com
echo    - API:      https://api.sinxdetect.movindu.com
echo.
echo Note: First startup may take 2-3 minutes for ML model loading.
echo.
goto end

:usage
echo Unknown mode: %MODE%
echo.
echo Usage: start-docker.bat [mode]
echo.
echo Modes:
echo   dev, development  - Start with hot-reload (frontend:5173, backend:8000)
echo   local             - Start combined container locally (port 3000)
echo   prod, production  - Start production build (port 80)
echo.
exit /b 1

:end
