# Fase 6 — Patchright E2E

Suite sequencial, própria e determinística para `SPEC_FINAL.md:232-285`.
O runner deriva a raiz deste arquivo, usa apenas `127.0.0.1:8420/8421`,
adquire `/tmp/llm-launcher-amd-e2e.lock` e nunca adota ou mata processos que
já estejam nas portas. O caminho crítico é executado antes dos cenários de
aba; todos os cenários implementados recebem `RunContext` e usam `GuardedAPI`.

## Preparação

O ambiente E2E vive em `~/.cache/llm-launcher-amd/e2e` (venv + Patchright +
Chromium) e as bibliotecas de execução do Chromium em
`~/.cache/llm-launcher-amd/e2e-libs`, para sobreviver a reboots do host.
Crie ou repare com o script versionado (idempotente):

```sh
scripts/setup-e2e.sh
```

Equivale a:

```sh
python3 -m venv ~/.cache/llm-launcher-amd/e2e
~/.cache/llm-launcher-amd/e2e/bin/pip install -r tests/e2e/requirements.txt
~/.cache/llm-launcher-amd/e2e/bin/patchright install chromium
```

O backend e a UI precisam estar construídos, mas o runner sobe seu próprio
backend em um processo/grupo filho somente durante a execução.
O preflight exige exatamente `patchright==1.61.2` via metadata; a versão
efetivamente encontrada é registrada em `environment.json`.

## Verificação sem efeitos

```sh
~/.cache/llm-launcher-amd/e2e/bin/python tests/e2e/run.py --help
~/.cache/llm-launcher-amd/e2e/bin/python tests/e2e/run.py --dry-run
~/.cache/llm-launcher-amd/e2e/bin/python -m compileall -q tests/e2e
```

## Execução

```sh
~/.cache/llm-launcher-amd/e2e/bin/python tests/e2e/run.py
```

O download HuggingFace externo é opt-in e não roda no caminho padrão:

```sh
~/.cache/llm-launcher-amd/e2e/bin/python tests/e2e/run.py --external-hf
```

O runner mantém um backend próprio e, para MCP, reinicia esse serviço em um
ciclo transitório: primeiro com MCP local habilitado e depois com MCP
desabilitado. O fixture MCP é local, versionado e sem rede; nenhum MCP remoto
é permitido. Settings adiciona a raiz E2E apenas durante a execução e o
snapshot restaura o estado persistente no cleanup.

Cada run escreve `logs/fase6-e2e/<run-id>/` com `backend.log`,
`environment.json`, `checklist.json`, `CHECKLIST.md` e screenshots. Em
qualquer falha o caminho crítico falha rápido, os demais itens ficam
`NÃO VERIFICADO`, e o `finally` para o grupo próprio, verifica GGUFs,
restaura bytes/existência/modo dos arquivos de estado e remove a raiz de
download somente quando o sentinel confere.

O modelo permitido para launch/delete é literalmente
`runtime/fase4-models/qwen2.5-1.5b-instruct-q4_k_m.gguf`; nenhum 27B/35B pode
passar pelo guard. O seed, manifest, logs existentes e estado persistente não
são alterados pela validação `--dry-run`.

## Contratos de segurança

Configs owned são registradas somente após uma resposta bem-sucedida de
`POST /api/configs` e usam o namespace da run (`e2e-...-<run-id>`). O método
`MutationGuard.register_launch(launch_id, config_ids)` aceita IDs opacos reais,
mas associa cada launch a configs owned; cancel/restart não possuem bypass.
O cleanup cancela apenas esses launches, aguarda `/api/launches=[]` e a porta
8421 livre antes de parar o backend. Em seguida restaura o estado por arquivos
temporários com `fsync`/`os.replace`, valida hashes/modos e grava
`cleanup.json`; qualquer erro invalida o Gate 6.
O backend recebe `LLM_LAUNCHER_HOST=127.0.0.1`,
`LLM_LAUNCHER_LLAMA_HOST=127.0.0.1` e `LLM_LAUNCHER_ALLOW_REMOTE_MCP=0`;
variáveis fora da allowlist são recusadas.
