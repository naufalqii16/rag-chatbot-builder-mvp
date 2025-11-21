#!/bin/bash
# ============================================================
# Start Qdrant Server with Web UI (Linux/Mac)
# ============================================================
# This script starts Qdrant in Docker with web UI on port 6333
#
# Prerequisites: Docker must be installed
# Web UI: http://localhost:6333/dashboard
# ============================================================

echo ""
echo "========================================"
echo "Starting Qdrant Server with Web UI"
echo "========================================"
echo ""

# Pull latest Qdrant image
echo "[1/2] Pulling Qdrant Docker image..."
docker pull qdrant/qdrant

echo ""
echo "[2/2] Starting Qdrant server..."
echo ""

# Start Qdrant with persistent storage
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/vectorstore/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

echo ""
echo "========================================"
echo "Qdrant Server Started!"
echo "========================================"
echo ""
echo "Web UI: http://localhost:6333/dashboard"
echo "API:    http://localhost:6333"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
