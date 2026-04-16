@echo off
REM EcoMarineAI -- unified startup script (Windows, Docker Desktop / WSL2).
REM
REM Usage:
REM     scripts\start.bat            -- dev compose (docker-compose.yml)
REM     scripts\start.bat prod       -- production compose (docker-compose.prod.yml)
REM
REM Requirements:
REM     - Docker Desktop running (WSL2 backend recommended)
REM     - curl.exe available (ships with Windows 10 1803+)
REM     - bash.exe via WSL or Git Bash (needed for download_models.sh)

setlocal enableextensions enabledelayedexpansion

set "MODE=%1"
if "%MODE%"=="" set "MODE=dev"

set "COMPOSE_FILE=docker-compose.yml"
if /I "%MODE%"=="prod" set "COMPOSE_FILE=docker-compose.prod.yml"

REM Jump to repo root (script is in <repo>\scripts)
pushd "%~dp0\.." || (
    echo [err] Cannot cd to repo root
    exit /b 1
)

echo [start] Checking Docker daemon...
where docker >nul 2>&1
if errorlevel 1 (
    echo [err] docker CLI not found. Install Docker Desktop first.
    popd & exit /b 1
)
docker info >nul 2>&1
if errorlevel 1 (
    echo [err] Docker daemon not reachable. Start Docker Desktop and retry.
    popd & exit /b 1
)

REM ------------------------------------------------------------------
REM Model weights: reuse scripts\download_models.sh via bash/WSL.
REM ------------------------------------------------------------------
if "%SKIP_DOWNLOAD%"=="1" (
    echo [start] SKIP_DOWNLOAD=1 -- skipping model download.
    goto :compose_up
)

set "MODEL_FILE=whales_be_service\src\whales_be_service\models\efficientnet_b4_512_fold0.ckpt"
if exist "%MODEL_FILE%" (
    echo [start] Model weights already present -- skipping download.
    goto :compose_up
)

echo [start] Downloading model weights...
where bash >nul 2>&1
if errorlevel 1 (
    echo [warn] bash.exe not found. Install Git for Windows or WSL, or set SKIP_DOWNLOAD=1.
    echo [warn] Continuing without weights -- backend may fail to start.
) else (
    bash scripts/download_models.sh
    if errorlevel 1 (
        echo [err] Model download failed.
        popd & exit /b 2
    )
)

:compose_up
echo [start] Launching Docker Compose stack (%COMPOSE_FILE%)...
docker compose -f %COMPOSE_FILE% up -d --remove-orphans
if errorlevel 1 (
    echo [err] 'docker compose up' failed. Inspect with:
    echo        docker compose -f %COMPOSE_FILE% logs
    popd & exit /b 3
)

echo [start] Waiting for http://localhost:8000/health ...
set /a ATTEMPTS=0
:health_loop
set /a ATTEMPTS+=1
curl.exe -fsS http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    echo [start] Backend is healthy.
    goto :done
)
if %ATTEMPTS% GEQ 90 (
    echo [err] Health check timed out after ~180s.
    echo [err] Inspect logs: docker compose -f %COMPOSE_FILE% logs backend
    popd & exit /b 4
)
timeout /t 2 /nobreak >nul
goto :health_loop

:done
echo [start] Stack is up:
echo         backend  -^> http://localhost:8000  (docs: /docs, metrics: /metrics)
echo         frontend -^> http://localhost:8080
echo [start] Stop with: docker compose -f %COMPOSE_FILE% down
popd
endlocal
exit /b 0
