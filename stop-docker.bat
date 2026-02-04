@echo off
REM Docker stop script for sinXdetect application on Windows

echo.
echo Stopping sinXdetect Application...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running.
    exit /b 1
)

REM Parse command line arguments
set MODE=%1
set CLEAN=%2
if "%MODE%"=="" set MODE=all

if /i "%MODE%"=="clean" goto cleanall
if /i "%MODE%"=="--clean" goto cleanall
if /i "%MODE%"=="prod" goto production
if /i "%MODE%"=="production" goto production
if /i "%MODE%"=="dev" goto development
if /i "%MODE%"=="development" goto development
if /i "%MODE%"=="local" goto local
goto stopall

:cleanall
echo Stopping ALL services and removing volumes...
docker compose down -v --remove-orphans 2>nul
docker compose -f docker-compose.dev.yml down -v --remove-orphans 2>nul
docker compose -f docker-compose.prod.yml down -v --remove-orphans 2>nul
echo.
echo All services stopped and volumes removed!
goto end

:production
echo Stopping PRODUCTION services...
if /i "%CLEAN%"=="clean" (
    docker compose -f docker-compose.prod.yml down -v --remove-orphans 2>nul
) else (
    docker compose -f docker-compose.prod.yml down --remove-orphans 2>nul
)
echo.
echo Production services stopped!
goto end

:development
echo Stopping DEVELOPMENT services...
if /i "%CLEAN%"=="clean" (
    docker compose -f docker-compose.dev.yml down -v --remove-orphans 2>nul
) else (
    docker compose -f docker-compose.dev.yml down --remove-orphans 2>nul
)
echo.
echo Development services stopped!
goto end

:local
echo Stopping LOCAL services...
if /i "%CLEAN%"=="clean" (
    docker compose down -v --remove-orphans 2>nul
) else (
    docker compose down --remove-orphans 2>nul
)
echo.
echo Local services stopped!
goto end

:stopall
echo Stopping ALL services...
docker compose down --remove-orphans 2>nul
docker compose -f docker-compose.dev.yml down --remove-orphans 2>nul
docker compose -f docker-compose.prod.yml down --remove-orphans 2>nul
echo.
echo All services stopped!
echo.
echo To remove volumes as well, run: stop-docker.bat clean
goto end

:end
echo.
echo To restart, run: start-docker.bat [dev^|local^|prod]
echo.
