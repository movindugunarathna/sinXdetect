@echo off
REM Quick setup script for LIME integration (Windows)

echo ==================================================
echo   LIME Integration Setup for SinBERT Classifier
echo ==================================================
echo.

REM Check if we're in the backend directory
if not exist "app.py" (
    echo Error: Please run this script from the backend directory
    exit /b 1
)

echo Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install dependencies
    exit /b 1
)

echo.
echo Dependencies installed successfully
echo.

echo Running integration tests...
python test_lime_integration.py

if errorlevel 1 (
    echo Tests failed, but you can still try running the server
)

echo.
echo ==================================================
echo   Setup Complete!
echo ==================================================
echo.
echo Next steps:
echo   1. Start the server: python app.py
echo   2. Test the API: python example_explanation.py
echo   3. View docs: http://localhost:8000/docs
echo.
echo Available endpoints:
echo   - POST /classify - Fast text classification
echo   - POST /classify-batch - Batch classification
echo   - POST /explain - LIME explanation with highlighting
echo.
pause
