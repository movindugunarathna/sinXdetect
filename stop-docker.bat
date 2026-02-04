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
set CLEAN=%1

if /i "%CLEAN%"=="clean" goto clean
if /i "%CLEAN%"=="--clean" goto clean
goto normal

:clean
echo Stopping services and removing volumes...
docker compose down -v --remove-orphans
docker compose -f docker-compose.dev.yml down -v --remove-orphans 2>nul
echo.
echo Services stopped and volumes removed!
goto end

:normal
echo Stopping services...
docker compose down --remove-orphans
docker compose -f docker-compose.dev.yml down --remove-orphans 2>nul
echo.
echo Services stopped!
echo.
echo To remove volumes as well, run: stop-docker.bat clean
goto end

:end
echo.
echo To restart, run: start-docker.bat
echo.
