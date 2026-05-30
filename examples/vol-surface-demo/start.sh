#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "==> Building @ngv/opx WASM (browser target)…"
(cd "$REPO_ROOT/bindings/wasm" && wasm-pack build --target web --out-dir pkg-web)

echo "==> Installing frontend dependencies…"
(cd "$(dirname "$0")/frontend" && npm install)

echo "==> Installing backend dependencies (uv)…"
(cd "$(dirname "$0")/backend" && uv sync)

echo "==> Starting FastAPI backend on :8000…"
(cd "$(dirname "$0")/backend" && uv run uvicorn main:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

echo "==> Starting Vite dev server on :5173…"
(cd "$(dirname "$0")/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl-C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
