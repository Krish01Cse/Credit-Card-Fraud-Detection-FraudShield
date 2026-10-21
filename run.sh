#!/bin/bash
cd "$(dirname "$0")"
export PATH="$PATH:/Users/krish_01/Library/Python/3.9/bin"

echo "🚀 Starting FastAPI Backend at http://127.0.0.1:8000 ..."
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "⚛️ Starting React Frontend at http://127.0.0.1:5173 ..."
cd frontend && npm run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
