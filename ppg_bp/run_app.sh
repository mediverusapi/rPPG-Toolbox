#!/bin/bash

# rPPG Blood Pressure Estimation - Streamlit App Runner
# This script sets up and runs the web application

echo "🩺 Starting rPPG Blood Pressure Estimation App"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Please run this script from the ppg_bp directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed or not in PATH"
    exit 1
fi

# Install requirements if they don't exist
echo "📦 Checking dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..."
    pip3 install -r requirements.txt
else
    echo "⚠️  requirements.txt not found, installing basic dependencies..."
    pip3 install streamlit opencv-python torch numpy pandas matplotlib plotly scipy
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit installation failed"
    exit 1
fi

echo "✅ Dependencies installed successfully"
echo ""
echo "🚀 Starting Streamlit app..."
echo "The app will open in your default browser automatically."
echo "If it doesn't open, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run the Streamlit app
streamlit run streamlit_app.py --server.address localhost --server.port 8501 