#!/bin/bash
# Run both backend and frontend for local development
# Usage: ./scripts/dev.sh

set -e

cd "$(dirname "$0")/.."

echo "Starting Kent Hologram development servers..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""

# Kill background processes on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start backend
echo "[Backend] Starting FastAPI server..."
uv run uvicorn web.backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo "[Frontend] Starting Next.js dev server..."
cd web/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "Both servers running. Press Ctrl+C to stop."
echo ""

# Wait for either process to exit
wait
