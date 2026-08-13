#!/usr/bin/env bash
# setup-e2e.sh — cria/repara o ambiente E2E persistente em
# ~/.cache/llm-launcher-amd/e2e (venv + patchright==1.61.2 + chromium) e as
# bibliotecas de execução do chromium em ~/.cache/llm-launcher-amd/e2e-libs.
# Idempotente: se o venv existe com patchright 1.61.2, pula a instalação.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${HOME}/.cache/llm-launcher-amd"
VENV_DIR="${CACHE_DIR}/e2e"
LIBS_DIR="${CACHE_DIR}/e2e-libs"
VENV_PY="${VENV_DIR}/bin/python"
LIBS_TARGET="${LIBS_DIR}/root/usr/lib/x86_64-linux-gnu"

mkdir -p "$CACHE_DIR"

# --- venv + patchright -------------------------------------------------------
needs_setup=0
if [ ! -x "$VENV_PY" ]; then
    needs_setup=1
elif ! "$VENV_PY" -c 'import importlib.metadata as m; assert m.version("patchright") == "1.61.2"' >/dev/null 2>&1; then
    needs_setup=1
fi

if [ "$needs_setup" -eq 1 ]; then
    echo "criando venv E2E em ${VENV_DIR} ..."
    # Preferir `python3 -m venv`; em hosts Debian/Ubuntu sem o pacote
    # python3-venv, cair para `python3 -m virtualenv` quando disponível.
    if ! python3 -m venv "$VENV_DIR" 2>"$CACHE_DIR/venv.err"; then
        echo "  python3 -m venv indisponível (ensurepip ausente); usando python3 -m virtualenv ..."
        python3 -m virtualenv "$VENV_DIR"
    fi
    rm -f "$CACHE_DIR/venv.err"
    "${VENV_DIR}/bin/pip" install -q -r "$REPO_ROOT/tests/e2e/requirements.txt"
    "${VENV_DIR}/bin/patchright" install chromium
else
    echo "venv E2E já existe com patchright 1.61.2; pulando."
fi

# --- bibliotecas do chromium (LD_LIBRARY_PATH do runner) ----------------------
# Não depende de /tmp (efêmero). Extrai .debs versionados do repo ou usa
# patchright install-deps como fallback.
if [ ! -d "$LIBS_TARGET" ]; then
    mkdir -p "$LIBS_DIR"
    if ls "$REPO_ROOT"/tests/e2e/scenarios/*.deb >/dev/null 2>&1; then
        echo "extraindo .debs versionados de tests/e2e/scenarios ..."
        for deb in "$REPO_ROOT"/tests/e2e/scenarios/*.deb; do
            dpkg-deb -x "$deb" "$LIBS_DIR/root"
        done
    elif command -v patchright >/dev/null 2>&1; then
        echo "tentando patchright install-deps ..."
        "${VENV_DIR}/bin/patchright" install-deps chromium 2>/dev/null || true
    else
        echo "AVISO: nem .debs nem patchright install-deps disponíveis." >&2
        echo "  As libs do chromium podem faltar. Instale libasound2, libgbm1," >&2
        echo "  libwayland-server0 manualmente se o E2E falhar ao lançar o browser." >&2
    fi
else
    echo "bibliotecas E2E já presentes em ${LIBS_DIR}; pulando."
fi

echo
echo "Ambiente E2E pronto. Python: ${VENV_PY}"
