#!/usr/bin/env bash
# start-backend.sh — sobe o backend em loopback, desacoplado do shell,
# idempotente: se 8420 já responde, não sobe um segundo processo.
# Log em logs/backend.log. Uso diário no host do launcher.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if curl -fsS --max-time 2 http://127.0.0.1:8420/api/options >/dev/null 2>&1; then
    echo "backend já responde em http://127.0.0.1:8420; nada a fazer."
    exit 0
fi

if [ ! -x app/.venv/bin/python ]; then
    echo "ERRO: app/.venv/bin/python ausente; construa o app primeiro (ver docs/WORKFLOW.md)" >&2
    exit 1
fi

mkdir -p logs

env -u LLM_LAUNCHER_HOST -u LLM_LAUNCHER_LLAMA_HOST setsid nohup app/.venv/bin/python app/api/server.py > logs/backend.log 2>&1 &

echo "backend iniciado (pid $!) em loopback; log em logs/backend.log"

for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8420/api/options >/dev/null 2>&1; then
        echo "backend pronto: http://127.0.0.1:8420"
        exit 0
    fi
    sleep 1
done

echo "AVISO: backend iniciado mas ainda não respondeu em 8420; veja logs/backend.log" >&2
exit 1
