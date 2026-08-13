# Registro reconstruído de build e execução

Este arquivo é uma reconstrução narrativa baseada em evidências versionadas.
`SPEC_llm-launcher-amd-fork.md` não existe no clone nem no histórico; não houve
rename físico desse arquivo. As instruções da execução foram a autoridade
registrada em `logs/DECISIONS.log`.

## Decisões e causas-raiz relevantes

- Não instalar com `sudo` sem autenticação e não repetir um bloqueio idêntico;
  a retomada só ocorreu quando o toolchain foi disponibilizado sem sudo.
- Usar telemetria AMD por sysfs, execução por HIP e manter
  `HIP_VISIBLE_DEVICES` sob controle do ambiente, em vez de presumir
  NVIDIA/CUDA ou uma ordem de GPUs.
- Corrigir headers C++ no caminho do GCC selecionado, em vez de atribuir
  `cmath not found` a rocWMMA.
- Tratar `8421` como porta reservada do `llama-server`; `1234` ficou apenas
  como uso legado observado no fluxo anterior.
- Considerar readiness válido somente após `/health` e manter isolamento por
  PGID/SID para que Stop não atingisse o backend ou processos externos.
- Separar resultados brutos do harness da adjudicação e preservar ambos; não
  transformar notas em PASS sem evidência.

## Fases 0–9

### Fase 0 — preparação

O fork foi clonado em seu diretório isolado, a branch `feature/amd-web` foi
criada e `app/.venv` recebeu as dependências. A verificação `sudo -n true`
exigiu senha, então a instalação privilegiada não foi feita nessa tentativa.

### Fase 1 — base web headless

O frontend passou a ser servido pelo FastAPI; Electron e seus resíduos de
workflow foram removidos. O backend foi validado com `/` e `/api/options`.
Após a remediação, produção usa same-origin, desenvolvimento usa proxy Vite e
CORS é opt-in local. O backend web ocupou a faixa a partir de `8420`.

### Fase 2 — AMD/HIP no launcher

Foi implementada telemetria AMD por sysfs, enumeração dinâmica de placas,
agregação de VRAM e a rota `/api/gpu`. Os defaults de `llama-server` e
`llama-cli` passaram a apontar para `vendor/llama.cpp/build/bin`; o teste
registrado encontrou uma AMD com 24.560 MiB.

### Fase 3 — toolchain e build do llama.cpp

A primeira tentativa foi bloqueada pela ausência de `hipcc`, `hipconfig` e
clang ROCm, combinada com `sudo` sem autenticação não interativa. Na retomada,
o toolchain ficou disponível sem uso de sudo e `vendor/llama.cpp` foi clonado
no commit `6a32c29a746a2e44de463de647f9f6661eb5086b`.

O erro real da primeira configuração HIP foi que o clang selecionou **GCC12
sem os headers C++ correspondentes**, produzindo `cmath not found`. **Não era
um erro de rocWMMA.** A correção foi indicar
`--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11` via
`CMAKE_HIP_FLAGS`. O build passou com `GGML_HIP=ON` e `GPU_TARGETS=gfx1100`.

### Fase 4 — inferência AMD mínima

Um Qwen GGUF foi baixado em diretório isolado, validado por tamanho e SHA-256,
e carregado no `llama-server`. A porta `8421` estava livre, `/health` retornou
HTTP 200 e a VRAM subiu de 26 para 2.320 MiB. O log confirmou ROCm e offload
de `29/29` camadas.

### Fase 5 — contrato HTTP

O endpoint OpenAI-compatible `/v1/chat/completions` respondeu HTTP 200, com
JSON válido, conteúdo `Olá` e `finish_reason=stop`. O backend web ficou em
`8420` e o `llama-server` em `8421` durante a validação.

### Fase 6 — UI AMD

A aba NVIDIA foi substituída por AMD GPU, preservando a estrutura da UI e
expondo VRAM agregada e por placa. Testes da API, teste focado AMD e build do
frontend passaram. A telemetria não fornecia uso/busy nem temperatura; a UI
registrou essa ausência sem inventar valores.

### Fase 7 — fluxo Launch/Stop

O primeiro ciclo de browser teve bloqueios de executável, bibliotecas e
locators; o ciclo seguinte revelou um problema real de portabilidade. O fluxo
legado lançava em **1234**, porta convencional de outro contexto, enquanto o
contrato atual reserva **8421** para o `llama-server`. Esse conflito impediu a
prova de readiness.

Também houve um problema de caminho: o arquivo de settings relativo ao cwd
fazia o backend localizar o modelo, mas a UI inicialmente mostrava zero. A
correção tornou o caminho absoluto na raiz do fork e o modelo passou a aparecer
após a estabilização do backend.

O readiness anterior era falso/não autoritativo: o estado da UI podia indicar
launch sem uma confirmação válida do servidor. A correção passou a fazer
preflight da porta e probe de `/health`; `8421` foi reservada e o processo
lançado passou a ter PGID/SID próprios. O Stop passou a validar ownership e a
encerrar somente o grupo/sessão da própria execução, preservando o backend.
O retry 2 confirmou Launch, readiness e Stop. A UI não contém chat; o fato foi
registrado como `NO_CHAT_UI_IN_APP`.

### Fase 8 — smoke test de produção

Ornith, Fable non-MTP e Fable MTP foram executados com as configurações e
resultados consolidados em [BENCHMARKS](BENCHMARKS.md). O resultado global foi
`PASSED_WITH_NOTES`; falhas brutas do harness e a adjudicação foram mantidas
separadas. Os processos foram encerrados pela UI, Ollama e Immich não foram
tocados, e a VRAM voltou ao baseline.

### Fase 9 — endurecimento e auditoria

O hardening alterou defaults de bind para loopback salvo override explícito,
adicionou contenção de paths, reforçou o boundary do MCP, rejeitou argumentos
inseguros e ajustou o surface de release. A auditoria registrou `PASS_WITH_NOTES`:
Gitleaks ficou limpo e os testes/build passaram, mas Trivy e OSV-Scanner não
estavam disponíveis, e os advisories npm permaneceram sem correção. Foi criada
a tag local `v0.1.0-amd-web`; não houve push.

## Fase 10 — fechamento documental

Este registro, [BENCHMARKS](BENCHMARKS.md) e [NEXT_STEPS](NEXT_STEPS.md)
consolidam o estado para uma LLM que recebeu apenas o clone. O fechamento não
reabre gates nem transforma gaps em tarefas concluídas. Para a evidência
primária, consulte evidência registrada no repositório privado (logs/STATUS.json),
evidência registrada no repositório privado (logs/DECISIONS.log), o relatório
(evidência registrada no repositório privado (logs/MORNING_REPORT.md)),
os logs `fase0.log`–`fase8.log` e a auditoria da Fase 9 (evidência registrada no
repositório privado (logs/FASE9_SECURITY_AUDIT.md)).
