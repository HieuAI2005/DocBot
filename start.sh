#!/bin/bash

# DocBot Startup Script
# This script starts both backend and frontend servers

echo "🚀 Starting DocBot Web Application..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source /home/hiwe/environment/nlp_viettel/bin/activate

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20 2>/dev/null || nvm use default

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
fi

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
cd /home/hiwe/project/DocBot
pip install -q -r requirements.txt
pip install -q -r backend/requirements.txt

# Start backend server in background
echo -e "${GREEN}Starting backend server on http://localhost:8000${NC}"
export PYTHONPATH=/home/hiwe/project/DocBot:$PYTHONPATH
cd /home/hiwe/project/DocBot
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check if npm is installed for frontend
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed. Please install Node.js and npm first.${NC}"
    echo "You can install with: sudo apt install npm"
    kill $BACKEND_PID
    exit 1
fi

# Install frontend dependencies and start
echo -e "${BLUE}Installing frontend dependencies...${NC}"
cd /home/hiwe/project/DocBot/frontend
npm install

echo -e "${GREEN}Starting frontend server on http://localhost:5173${NC}"
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo -e "${GREEN}✅ DocBot is running!${NC}"
echo ""
echo "📍 Frontend: http://localhost:5173"
echo "📍 Backend API: http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Handle Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Wait for processes
wait
