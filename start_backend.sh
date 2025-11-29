#!/bin/bash

# Quick startup script for backend only
# Use this if you have Node.js version issues

echo "🚀 Starting DocBot Backend Server..."

# Activate virtual environment
source /home/hiwe/environment/nlp_viettel/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

# Set Python path
export PYTHONPATH=/home/hiwe/project/DocBot:$PYTHONPATH

# Navigate to project root
cd /home/hiwe/project/DocBot

echo "✓ Starting backend on http://localhost:8000"
echo "✓ API docs available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run backend server
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
