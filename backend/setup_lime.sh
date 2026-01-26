#!/bin/bash
# Quick setup script for LIME integration

echo "=================================================="
echo "  LIME Integration Setup for SinBERT Classifier"
echo "=================================================="
echo ""

# Check if we're in the backend directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    exit 1
fi

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✓ Dependencies installed successfully"
echo ""

echo "🧪 Running integration tests..."
python test_lime_integration.py

if [ $? -ne 0 ]; then
    echo "⚠️  Tests failed, but you can still try running the server"
fi

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Start the server: python app.py"
echo "  2. Test the API: python example_explanation.py"
echo "  3. View docs: http://localhost:8000/docs"
echo ""
echo "Available endpoints:"
echo "  • POST /classify - Fast text classification"
echo "  • POST /classify-batch - Batch classification"
echo "  • POST /explain - LIME explanation with highlighting"
echo ""
