# Próximos passos

Esta lista contém somente trabalho aberto sustentado pela evidência atual.

- **Multi-GPU:** validar em hardware a colocação (placement) de camadas e
  tensores e a configuração de split entre GPUs.
- **RX 7800 XT:** quando a placa chegar, refazer o build com
  `GPU_TARGETS='gfx1100;gfx1101'`.
- **GGUF multi-shard:** modelos divididos em vários arquivos `.gguf` não são
  suportados. "Shard faltando" é tratado como caso sem auto-degrade — o loop
  resiliente desiste ([runner.py](../app/api/core/runner.py#L530-L538)).
- **Peer remoto para `SETTINGS-06`:** o item E2E "remote settings POST returns
  403" fica `NÃO VERIFICADO` sem um peer remoto controlado na execução; o
  cenário não falsifica headers remotos ([system_ui.py](../tests/e2e/scenarios/system_ui.py#L187-L190)).
- **Advisories de dev (Vite/esbuild):** `npm audit --omit=dev` em `app/`
  resulta em **0 vulnerabilidades** (verificado em 2026-08-13). As contagens
  capturadas — 3 high + 1 moderate + 1 low (5 no total), conforme
  evidência registrada no repositório privado (logs/fase11/A.log) — são todas de dependências de
  desenvolvimento (`@babel/core`, `esbuild <=0.24.2`, `vite <=6.4.2` por
  herança do esbuild, `nanoid <=3.3.16`); corrigir o Vite exige
  `npm audit fix --force` (vite@8.2.1, breaking change).
- **`release.yml`:** consertar antes de reintroduzir GitHub Actions no público.
  O workflow atual instala só `requirements.txt`, mas a coleta do pytest exige
  `httpx2` de `requirements-dev.txt` (comentado no próprio
  [requirements-dev.txt](../app/requirements-dev.txt)); não tem gates de
  segurança (gitleaks/`npm audit`); e nunca foi exercitado com uma tag
  pushada — só a tag local `v0.1.0-amd-web` existe
  (evidência registrada no repositório privado (logs/FASE9_SECURITY_AUDIT.md)). Enquanto
  isso, o repo público não leva `.github/`.
- **PRs upstream:** as correções validadas neste fork (loopback por default,
  contenção de caminhos, hardening MCP) ainda não foram submetidas como PRs ao
  projeto origem (`RAFAEL-SILVASOUZA/llm-launcher`).
- **Scanners:** executar Trivy e OSV-Scanner; ambos permanecem pendentes.
- **LM Studio herdado — defeito de segurança:** no host validado, esse caminho
  está inativo porque o LM Studio não está instalado. Porém,
  [`app/api/server.py:1037`](../app/api/server.py) expõe `/api/lms/load` e o
  fallback em [`LaunchModal.tsx:137`](../app/src/components/LaunchModal.tsx)
  pode ativá-lo quando o LMS estiver instalado. O comando herdado em
  [`app/api/core/lms.py:73`](../app/api/core/lms.py) contém
  `--bind 0.0.0.0 --cors`, o defeito de segurança de abertura em todas as
  interfaces e habilitação de CORS que deve ser corrigido.
