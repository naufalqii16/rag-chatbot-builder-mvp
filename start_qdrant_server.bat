@echo off
REM ============================================================
REM Start Qdrant Server with Web UI
REM ============================================================
REM This script starts Qdrant in Docker with web UI on port 6333
REM
REM Prerequisites: Docker must be installed
REM Web UI: http://localhost:6333/dashboard
REM ============================================================

echo.
echo ========================================
echo Starting Qdrant Server with Web UI
echo ========================================
echo.

REM Pull latest Qdrant image
echo [1/2] Pulling Qdrant Docker image...
docker pull qdrant/qdrant

echo.
echo [2/2] Starting Qdrant server...
echo.

REM Start Qdrant with persistent storage
docker run -p 6333:6333 -p 6334:6334 ^
    -v %CD%\vectorstore\qdrant_storage:/qdrant/storage:z ^
    qdrant/qdrant

echo.
echo ========================================
echo Qdrant Server Started!
echo ========================================
echo.
echo Web UI: http://localhost:6333/dashboard
echo API:    http://localhost:6333
echo.
echo Press Ctrl+C to stop the server
echo.
