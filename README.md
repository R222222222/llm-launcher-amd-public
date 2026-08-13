# <img src="app/public/logo.png" width="42" alt="LLM Launcher logo" align="top"> LLM Launcher

![Plataforma](https://img.shields.io/badge/plataforma-Linux%20%2B%20AMD-555555.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933.svg?logo=nodedotjs&logoColor=white)
![Web headless](https://img.shields.io/badge/Web%20headless-React%20%2B%20FastAPI-47848F.svg)
![GPU](https://img.shields.io/badge/GPU-AMD%20%2F%20llama.cpp-ED1C24.svg)
![llama.cpp](https://img.shields.io/badge/motor-llama.cpp-orange.svg)
![Licença](https://img.shields.io/badge/licen%C3%A7a-PolyForm%20Noncommercial%201.0.0-blue.svg)

Launcher visual + CLI para rodar modelos GGUF locais com **llama.cpp**, com foco no
fork AMD/web e configuração portátil por ambiente.

> **Fluxo recomendado, do zero ao play:** siga o guia único
> [docs/WORKFLOW.md](docs/WORKFLOW.md), que cobre build, túnel SSH, seed,
> launch, MCP, OpenCode e encerramento limpo.

## Documentação do projeto

O [contexto do projeto](docs/PROJECT_CONTEXT.md) é o ponto de entrada para uma
LLM que acabou de clonar o repositório. Consulte também a
[arquitetura](docs/ARCHITECTURE.md), as [decisões](docs/adr/), os
[benchmarks](docs/BENCHMARKS.md), o [registro reconstruído de
build](docs/BUILD_LOG.md) e os [próximos passos](docs/NEXT_STEPS.md).

O projeto tem duas frentes que compartilham as mesmas configurações:

| Frente | O que é | Quando usar |
|---|---|---|
| **APP web** (`app/`) | Aplicação web headless (React + FastAPI) com grid de configurações, estimador de VRAM, download do HuggingFace, telemetria AMD e gerenciador opcional de servidores MCP | Uso visual no navegador |
| **CLI** (`models.py`) | Script interativo de terminal que monta e executa a linha de comando do `llama-server`/`llama-cli`, com auto-degrade quando o servidor cai | Terminal, automação, máquinas sem interface |

![Tela inicial](telas/tela%20inicial.png)

---

## Sumário

1. [Requisitos](#1-requisitos)
2. [Instalação do zero](#2-instalação-do-zero)
   - [2.1 Clonar este repositório](#21-clonar-este-repositório)
   - [2.2 Clonar e compilar o llama.cpp](#22-clonar-e-compilar-o-llamacpp)
   - [2.3 Instalar as dependências do Python](#23-instalar-as-dependências-do-python)
   - [2.4 Configurar os caminhos](#24-configurar-os-caminhos)
3. [A APP web](#3-a-app-web)
   - [3.1 Rodando em desenvolvimento](#31-rodando-em-desenvolvimento)
   - [3.2 Build e execução headless](#32-build-e-execução-headless)
   - [3.3 Arquitetura](#33-arquitetura)
   - [3.4 As telas](#34-as-telas)
4. [O CLI (models.py)](#4-o-cli-modelspy)
5. [Onde ficam os arquivos de configuração](#5-onde-ficam-os-arquivos-de-configuração)
6. [Auto-degrade: o que acontece quando o servidor cai](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai)
7. [Solução de problemas](#7-solução-de-problemas)
8. [Segurança e advisories conhecidos](#8-segurança-e-advisories-conhecidos)
9. [Licença](#9-licença)

---

## 1. Requisitos

Antes de começar, instale:

| Ferramenta | Para quê | Onde baixar |
|---|---|---|
| **Git** | clonar os repositórios | https://git-scm.com/downloads |
| **Python 3.10+** | backend da APP e o CLI | https://www.python.org/downloads/ |
| **Node.js 18+ (LTS)** | frontend Vite da APP web | https://nodejs.org/ |
| **CMake 3.24+** | compilar o llama.cpp | https://cmake.org/download/ |
| **Compilador C/C++** | build do llama.cpp para seu host/GPU | toolchain nativo da sua distribuição |

> 💡 O backend é portátil: use o toolchain e o backend GPU disponíveis no host.

> ⚠️ Neste host, o Node.js do sistema é v12.22.9. Para a APP, use Node.js 18+
> pelo **nvm**; use um shell de login ou carregue o nvm antes dos
> comandos do frontend: `source ~/.nvm/nvm.sh && nvm use 22`.
>
> Combinação testada neste clone: Node.js **v22.22.2** com npm **10.9.7**,
> ambos fornecidos pelo nvm.

---

## 2. Instalação do zero

### 2.1 Clonar este repositório

Em um shell POSIX:

```bash
git clone <fork-url> llm-launcher-amd
cd llm-launcher-amd
```

### 2.2 Clonar e compilar o llama.cpp

O launcher **não traz o llama.cpp junto** — ele executa os binários
`llama-server` / `llama-cli` que você compila. Ele conhece quatro *backends*
(builds diferentes):

| Backend | O que é | Obrigatório? |
|---|---|---|
| `vanilla` | llama.cpp oficial (upstream), com os parsers de chat/tool mais novos e MTP nativo | ✅ **Sim — comece por ele** |
| `mtp` | Não é um build separado: é o mesmo binário do `vanilla` lançado com `--spec-type draft-mtp` (decodificação especulativa para modelos com head MTP) | Vem de graça com o vanilla |
| `turbo` | Fork customizado ("turboquant") com tipos extras de KV cache `turbo2/3/4` (compressão agressiva para contexto longo) | Opcional — se não tiver, use os KV padrão (`f16`, `q8_0`…) |
| `custom` | **Qualquer outro build** — o fork que um modelo específico exige (arch nova ainda não mergeada, patch de tokenizer…). Não tem caminho fixo: você aponta a pasta dos binários em **Settings → Diretórios dos binários llama.cpp** (ou no menu Settings do CLI) e troca quando quiser experimentar outro | Opcional |

Sobre o `custom`: nada é assumido sobre o build. Os tipos de KV cache oferecidos saem do `--help` do binário que você apontou (então um fork com KV próprio aparece com os tipos dele), e o *speculative MTP* fica desligado — emitir `--spec-type draft-mtp` num fork desconhecido quebra o load e o auto-degrade não sabe remover essa flag. Se o build for exigente quanto a flags, combine `custom` com o checkbox **"llama.cpp decide (comando mínimo)"**.

**Passo a passo (build AMD/HIP reproduzível):**

```bash
# O default do fork espera este layout; Settings também aceita outro diretório.
mkdir -p vendor
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
git -C vendor/llama.cpp checkout 6a32c29a746a2e44de463de647f9f6661eb5086b
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build -j
```

Os binários esperados são `vendor/llama.cpp/build/bin/llama-server` e
`vendor/llama.cpp/build/bin/llama-cli` ([evidência do build](docs/BUILD_LOG.md#L46-L57)).
No host de referência, se o clang HIP não encontrar os headers GCC, a configuração
precisou de `-DCMAKE_HIP_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`;
isso é uma solução específica desse host, não um requisito geral
(evidência registrada no repositório privado (logs/fase3.log)). Para outro host AMD, mantenha as
flags HIP compatíveis com seu ROCm/toolchain e aponte a pasta em **Settings**.

Para **atualizar** o llama.cpp no futuro:

```bash
cd vendor/llama.cpp
git fetch --tags
git checkout 6a32c29a746a2e44de463de647f9f6661eb5086b
cd ../..
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build -j
```

### 2.3 Instalar as dependências do Python

```bash
# Ambiente Python da APP (FastAPI + uvicorn)
python3 -m pip install --user virtualenv
python3 -m virtualenv app/.venv
app/.venv/bin/python -m pip install -r app/requirements.txt

# Testes e ferramentas de desenvolvimento
app/.venv/bin/python -m pip install -r app/requirements-dev.txt

# CLI interativo
app/.venv/bin/python -m pip install questionary
```

> As versões em `app/requirements.txt` são **pinadas**. Faixas abertas
> (`>=`) já quebraram a suite de testes quando o `starlette` passou a
> exigir `httpx2`; não afrouxe sem rodar `pytest app/api/tests -q`.

### 2.4 Configurar os caminhos

O launcher precisa saber **duas coisas**: onde estão seus **modelos (.gguf)** e onde estão os **binários do llama.cpp**.

Existem dois lugares onde isso é definido:

1. **Defaults portáteis** — `app/api/core/constants.py` (APP) e o topo de
   `models.py` (CLI) usam o repositório e `Path.home()`:
   - Modelos: `runtime/models` relativo ao fork;
   - LM Studio CLI: `~/.lmstudio/bin/lms`;
   - llama.cpp: `vendor/llama.cpp/build/bin`.

   Os defaults podem ser sobrescritos sem editar código:

   ```bash
   export LLM_LAUNCHER_MODELS_DIR="$HOME/models"
   export LLM_LAUNCHER_LMS_PATH="$HOME/.lmstudio/bin/lms"
   export LLM_LAUNCHER_LLAMA_CPP_BIN="$PWD/vendor/llama.cpp/build/bin"
   ```

   `LLM_LAUNCHER_LLAMA_CPP_BIN` é o override de ambiente de `LLAMA_CPP_BIN` e
   funciona tanto na APP quanto no CLI. O caminho default de modelos é
   `runtime/models`; ele pode ser criado vazio e o scanner simplesmente retorna
   uma lista vazia enquanto ainda não houver `.gguf`.

   O arquivo `.env.example` documenta esses defaults sem segredos. Para usá-lo
   em um shell POSIX, copie-o e exporte suas variáveis (o launcher não carrega
   `.env` automaticamente):

   ```bash
   cp .env.example .env
   set -a; . ./.env; set +a
   ```

   Se seus caminhos forem outros, ajuste essas constantes **ou** use a opção 2 (recomendado):

2. **Tela Settings da APP** (ou modo `settings` do CLI) — grava um `app_settings.json` compartilhado entre APP e CLI, com:
   - `model_paths`: lista de pastas varridas em busca de `.gguf`;
   - `backend_paths`: pasta dos binários de cada backend (sobrescreve o default sem tocar no código). É também como o backend `custom` ganha um caminho — ele não tem default utilizável.

   ![Tela Settings](telas/tela%20de%20configurações.png)

   O que foi configurado na APP vale para o CLI e vice-versa — é o mesmo arquivo, inclusive para os `backend_paths`.

> 📁 **Organização dos modelos:** o scanner procura `.gguf` recursivamente dentro das pastas cadastradas. A convenção `dono/repositorio/quant/arquivo.gguf` (a mesma do LM Studio e do download interno da APP) deixa a listagem mais legível.

---

## 3. A APP web

### 3.1 Rodando em desenvolvimento

```bash
cd app
npm ci              # primeira vez
npm start           # terminal 1: backend headless usando app/.venv
```

Em outro terminal, rode `npm run dev` para o Vite em `127.0.0.1:5173`. O proxy
encaminha `/api` para `http://127.0.0.1:8420` por padrão. Para um backend
remoto de desenvolvimento, opte explicitamente por:

```bash
# TAILSCALE_IP=<tailscale-ip>
: "${TAILSCALE_IP:?defina TAILSCALE_IP antes de iniciar}"
VITE_DEV_API_TARGET="http://${TAILSCALE_IP}:8420" npm run dev
```

Há dois overrides independentes: `LLM_LAUNCHER_HOST` controla onde o backend
FastAPI escuta e `LLM_LAUNCHER_LLAMA_HOST` controla o `--host` do
`llama-server`. Ambos usam `127.0.0.1` por padrão; mudar um não muda o outro.
Para publicar voluntariamente ambos em uma interface Tailscale, configure os
dois com o endereço do seu ambiente:

```bash
# TAILSCALE_IP=<tailscale-ip>
: "${TAILSCALE_IP:?defina TAILSCALE_IP antes de iniciar}"
export LLM_LAUNCHER_HOST="$TAILSCALE_IP"
export LLM_LAUNCHER_LLAMA_HOST="$TAILSCALE_IP"
npm start
```

`PYTHON` continua disponível para escolher outro interpretador. CORS é
opt-in somente para desenvolvimento (`LLM_LAUNCHER_DEV_CORS=1`), limitado a
`localhost:5173`/`127.0.0.1:5173`; em produção, o frontend servido pelo próprio
FastAPI usa `window.location.origin` (same-origin), sem `VITE_API_BASE`.

### 3.2 Build e execução headless

```bash
cd app
npm run build
npm start
```

O build gera `app/dist`, servido pelo FastAPI no host configurado em runtime e
na primeira porta livre a partir de **8420**, pulando **8421**, reservada ao
`llama-server` ([server.py](app/api/server.py#L1160-L1181)). Assim, **8420** é o
default somente quando está livre; o backend pode escolher uma porta posterior.
O seed canônico chama fixamente `127.0.0.1:8420`, portanto essa porta precisa
estar livre antes de iniciar o backend ([seed-profiles.py](scripts/seed-profiles.py#L17-L24)).
Em produção, o frontend usa sempre `window.location.origin`,
isto é, a API é same-origin. Abra o endereço local ou Tailscale explicitamente
configurado no seu ambiente.

**Releases no GitHub:** o repositório tem um workflow de CI (`.github/workflows/release.yml`) que faz tudo isso sozinho. Basta criar e enviar uma tag de versão:

```bash
git tag v0.2.0
git push origin v0.2.0
```

O GitHub Actions builda o bundle web num runner Linux, compacta `dist/` em
`LLM-Launcher-v0.2.0-web.tar.gz` e publica uma **Release**. A versão do
`package.json` é sincronizada com a tag automaticamente durante o build.

### 3.3 Arquitetura

```
app/
├── scripts/       # start-web.mjs: inicia o backend pela venv
├── src/          # frontend React + Tailwind (Vite)
│   └── components/   # uma página por aba + modais
└── api/          # backend FastAPI
    └── core/     # builder do comando, runner com auto-degrade, estimador
                  # de VRAM, scanner de GGUF, download HF, telemetria AMD, MCP
```

O frontend conversa com o backend por REST + SSE (eventos de launch e de download em tempo real). O backend é quem executa o `llama-server` — se a janela for fechada com um servidor rodando, ao reabrir a APP ela **reanexa** o processo órfão e volta a oferecer o botão de Stop.

### 3.4 As telas

#### Cabeçalho (presente em todas as telas)

Na faixa superior ficam as barras de **VRAM** e **RAM** em tempo real (atualizam a cada 2 s) e os *badges* dos backends — verde quando o binário foi encontrado no caminho configurado. Ao lado, o botão **refresh** recarrega tudo.

#### Aba Configs — a tela inicial

![Tela inicial — grid de configs](telas/tela%20inicial.png)

É o coração da APP: um grid com as configurações de launch salvas. Cada linha é um par **modelo + backend + parâmetros**:

- **Bolinha de status** (coluna `•`): estimativa de memória da config — 🟢 cabe na VRAM com folga, 🟡 apertado, 🔴 não cabe. Calculada em lote ao abrir a tela.
- **Colunas**: backend, contexto (CTX), tipo de KV cache, camadas na GPU (`-ngl`), camadas MoE na CPU (`-ncmoe`), slots paralelos (`-np`), estado do thinking/reasoning e flag de visão (mmproj).
- **Filtro e chips** no topo filtram por texto ou por backend (`todos / turbo / vanilla / mtp / custom`).
- **Ações por linha**: ✏️ editar, ⧉ duplicar, 🚀 **launch**, 🗑️ excluir.
- **Checkboxes à esquerda + launch múltiplo (modo router)**: marque várias configs e suba todas de uma vez — o launcher gera um preset para o `llama-server`, que carrega as configs com **load-on-startup** na porta única **8421**; o cliente escolhe o modelo pelo campo `model` da requisição. Só é permitido **um launch ativo por vez** (tentar outro retorna erro 409). [Fonte: `server.py`](app/api/server.py#L648-L706).
- **+ nova** abre o editor com uma config em branco.

Enquanto houver um servidor no ar, a tela mostra os controles de **Stop/parar**,
**Restart** e **abrir logs**.

#### Editor de configuração (modal "Editar configuração")

![Setup do modelo — parte 1](telas/tela%20de%20setup%20do%20modelo%20-%2001.png)

Abre ao criar/editar/duplicar uma config. À esquerda, os parâmetros; à direita, dois painéis que **atualizam ao vivo** conforme você mexe:

- **Estimativa**: barras de VRAM/RAM previstas para essa config (pesos + KV cache + compute), com *breakdown* expansível e metadados do GGUF (camadas, embedding, heads, arquitetura). Considera a VRAM já ocupada pelo sistema.
- **Comando**: a linha de comando exata do `llama-server` que será executada — dá para copiar e rodar manualmente se quiser.

Campos da primeira parte:

- **Modelo (.gguf)** — dropdown com tudo que o scanner achou nas pastas configuradas.
- **llama.cpp decide (comando mínimo)** — checkbox ao lado do dropdown. Marcado, o comando não recebe flags de tuning: contexto, `-ngl`, KV, threads, batch, samplers, reasoning e speculative ficam nos defaults do llama.cpp. O launcher ainda pode incluir, quando configurados, `--mmproj`, `--mcp-servers-config` e `--verbose`, além dos argumentos de identificação e HTTP ([builder.py — modo auto](app/api/core/builder.py#L49-L86)). No modo router, cada seção preserva `model`, `mmproj` e `verbose` quando aplicáveis e sempre recebe `load-on-startup`; configs com MCP são rejeitadas pelo endpoint router ([builder.py](app/api/core/builder.py#L313-L340), [L410-L416](app/api/core/builder.py#L410-L416) e [server.py](app/api/server.py#L661-L670)). Os demais campos continuam salvos na config e voltam a valer ao desmarcar. O [auto-degrade](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai) fica desligado, pois não há flags de tuning nossas para baixar.
- **Backend** — turboquant / vanilla / mtp (cards com descrição; desabilitado se o binário não existir).
- **Context window** e **Slots paralelos (`-np`)** — o launcher emite `--kv-unified`, então o contexto é um **pool único** que os slots dividem (é o que o LM Studio faz por padrão): o KV custa `contexto × 1`, e não `× slots`. Uma request sozinha pode usar o pool inteiro; várias simultâneas disputam. Se o build não tiver a flag, o `-c` volta a ser multiplicado pelos slots para não encolher a janela em silêncio — e aí o KV custa `× slots` mesmo.
- **KV cache (`--cache-type-k/v`)** — de `f16` (padrão) a quantizados (`q8_0`, `q4_0`…); os tipos `turbo2/3/4` só aparecem no backend turboquant.
- **Flash Attention (`-fa`)** — liga/desliga.
- **GPU / Offload (`-ngl`)** — quantas camadas vão para a GPU, com botão **"sugerir -ngl"** que calcula o máximo que cabe na VRAM. Para modelos MoE há também o `-ncmoe` (camadas de experts na CPU).
- **Avisos inteligentes** — ex.: ao detectar arquitetura híbrida SSM, sugere desligar o reasoning (evita travamento em loops agênticos); com 2 GPUs, expõe **split mode (`-sm`)** e **proporção (`-ts`)**.

![Setup do modelo — parte 2](telas/tela%20de%20setup%20do%20modelo%20-%2002.png)

Rolando o modal, a segunda parte:

- **Server (KV total, prompt cache, checkpoints)** — `cache-ram` (MiB de RAM para cache de prompts; acelera reuso de contexto) e `ctx-checkpoints` (restauração parcial de contexto editado).
- **Generation & misc** — `max-tokens (-n)` (proteção contra loop infinito), `batch (-b)`, `ubatch (-ub)` (pico de VRAM no prefill), threads de geração e de batch (`-t` / `-tb`).
- **Flags**: `--mlock`, `--verbose`, e o modo de execução — **server (HTTP)** ou **cli** (chat no terminal).

Botões: **Salvar** (só grava) ou **Salvar e launch** (grava e sobe o servidor imediatamente).

#### Modal de logs do launch

![Modal de logs](telas/modal%20de%20logs.png)

Abre automaticamente ao lançar. Mostra:

- Resumo da config no topo (backend, ctx, kv, ngl, np);
- O **comando executado** e o log do `llama-server` em tempo real (via SSE);
- Cada **tentativa** numerada — se o servidor cair na carga, o auto-degrade ajusta a config e tenta de novo (ver [seção 6](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai));
- Botões **Reiniciar** (mata e ressobe com a mesma config, mantendo a porta — o cliente reconecta sozinho), **Cancelar** (derruba o servidor) e **Esconder** (fecha o modal sem parar nada; reabra pelo botão de logs na aba Configs). O controle **Stop/parar** pertence à aba Configs/router; no modal, a ação é **Cancelar**.

Quando a carga termina com sucesso, o `llama-server` fica ouvindo por padrão em
`http://127.0.0.1:8421` com API compatível com OpenAI — aponte Claude Code,
Continue ou qualquer cliente OpenAI-compatible para ele. O host pode ser
substituído por `LLM_LAUNCHER_LLAMA_HOST`; **8421 é a porta padrão reservada ao
llama-server deste launcher** ([constants.py](app/api/core/constants.py#L31-L44)).

#### Aba Models

![Tela de modelos](telas/tela%20de%20modelos.png)

Inventário de tudo que o scanner encontrou nas pastas configuradas:

- **Alias + caminho relativo**, **tamanho em disco** (somando shards de modelos divididos);
- **Flags** detectadas pelos metadados do GGUF: 🧠 thinking/reasoning, ⚡ MTP, 🖼️ visão (mmproj);
- **delete** remove o `.gguf` do disco (e o mmproj associado), com confirmação e limpeza de pastas vazias.

O campo **filtrar…** busca por nome/caminho.

#### Aba Download

![Tela de download](telas/tela%20de%20downloads.png)

Baixa modelos direto do HuggingFace:

1. **Salvar em** — escolha a pasta de destino entre as cadastradas no Settings (é obrigatório ter ao menos uma).
2. **HuggingFace download** — cole a URL (`https://huggingface.co/dono/repo`) ou o id (`dono/repo`) e clique **inspecionar**: a APP lista os arquivos GGUF do repositório agrupados por quantização para você escolher qual baixar. Ou use o campo de **busca** (ex.: `qwen 30b moe coder`) para pesquisar repositórios GGUF no HF.
3. Durante o download: barra de progresso por arquivo com velocidade, suporte a modelos multi-shard, botão de cancelar. Pode **trocar de aba sem medo** — o download continua no backend e a aba Download mostra um spinner enquanto estiver ativo.
4. Ao final, a integridade é validada e a lista de modelos recarrega sozinha.
5. A APP busca o **`generation_config.json`** do repo (ou do `base_model`, quando é repo de quant) e grava um **`sampling.json`** ao lado do modelo — são os samplers que o *autor* publicou. Se o repo não publicar nenhum, a tela avisa.

#### Samplers por modelo

Temperatura, top-p, top-k, min-p e repeat-penalty **não são um preset único** — são resolvidos por modelo, e a APP mostra **de onde vieram**:

| Procedência | Significado |
|---|---|
| `valores do autor` | vieram do `generation_config.json` do repo (via `sampling.json`) |
| `preset reasoning` | o chat template do GGUF tem `<think>` → temp 0.6 / top-p 0.95, sem penalidade |
| `preset código` | template sem `<think>` → temp 0.3 / top-p 0.8 / repeat 1.05 (validado em A/B para agente de código) |
| `chute` | o GGUF não pôde ser lido — confira o card do modelo |
| `fixado por você` | você editou na seção **Sampling** do editor; nada sobrescreve |

Por que isso importa: em modelo de raciocínio, temperatura baixa e repeat-penalty **aumentam** o loop dentro do `<think>` e punem a repetição legítima de fórmula/identificador. Quem pensa e quem não pensa precisam de presets diferentes — e a detecção sai do **chat template**, nunca do nome do arquivo (`ornith-1.0-35b` é reasoning e não tem nenhuma palavra reveladora no nome).

#### Aba MCP

![Tela de MCPs](telas/tela%20de%20MCPs.png)

Esta aba é um supervisor simples de **processos locais**. Ela cadastra `cwd` e
`command`, inicia o comando com `shell=True` no diretório indicado, captura
stdout/stderr e permite ligar, desligar, editar e ver logs. **Não implementa o
protocolo MCP**: não faz JSON-RPC, `tools/list` nem conecta esses processos ao
`llama-server`.

Por segurança, MCP fica **desabilitado por padrão**. Habilite-o somente com
`LLM_LAUNCHER_ENABLE_MCP=1`; nesse modo a API aceita chamadas apenas em
loopback. `LLM_LAUNCHER_ALLOW_REMOTE_MCP=1` é um override explícito para um
ambiente controlado e deve ser tratado como perigoso: MCP executa comandos
arbitrários configurados pelo usuário, funcionando como um executor de
comandos remoto.

- Cada card mostra nome, diretório e comando, com botões **ligar/desligar**, 👁️ ver logs (stdout/stderr capturados), ✏️ editar e 🗑️ excluir.
- Se o processo cair, o status volta para desligado automaticamente.

![Cadastro de MCP](telas/tela%20MCPs%20-%20modal%20cadastro.png)

O cadastro pede só três campos: **Nome**, **Diretório (cwd)** e **Comando** (ex.:
`npm run dev`), executado via shell no diretório informado. Isso é supervisão
de processo, não uma definição MCP para o `llama-server`; o comando pode
executar código local e deve ser tratado como entrada privilegiada.

#### MCP do `llama-server` — Fase 3 revisada

O fluxo automatizado usa o campo de configuração `mcp_servers_config` e a opção
`--mcp-servers-config` do próprio `llama-server`. O arquivo apontado usa o
formato Cursor-compatible para servidores **stdio**; cada servidor precisa de
um `command` **absoluto** (não dependa do `PATH` da APP), por exemplo:

```json
{
  "mcpServers": {
    "exemplo": {
      "command": "/absolute/path/to/mcp-server",
      "args": ["--stdio"]
    }
  }
}
```

O exemplo neutro versionado está em [`config/mcp/servers.example.json`](config/mcp/servers.example.json).
O launcher aceita somente o caminho absoluto canônico `config/mcp/servers.json`,
ignorado pelo Git, e valida o schema stdio antes de salvar ou lançar. A migração
do arquivo runtime histórico é manual; o launcher não o copia nem remove.
O endpoint `/api/options` expõe apenas metadados não secretos desse arquivo em
`mcp_runtime_config` (`path`, `exists`, `valid`).

O JSON real é um arquivo local fora do versionamento e pode conter comandos,
argumentos e ambiente sensíveis; não o commite nem confunda o arquivo de
referência com uma configuração de produção. No startup, o `llama-server`
inicia esses comandos: habilitar MCP equivale a permitir execução de código
local. Use somente arquivos e binários confiáveis. Um JSON com apenas `url`
ou uma URL HTTP-stream não pertence a esse campo.

O caminho alternativo/histórico é a WebUI do `llama-server`: ela cadastra MCP
por URL/HTTP-stream e mantém servidor e aprovação no estado do navegador. Pode
ser usada manualmente, mas não é o fluxo automatizado da Fase 3 e não há
cadastro WebUI obrigatório. A opção `--ui-mcp-proxy`, associada a esse fluxo,
fica superada para a Fase 3 e não deve ser usada. A aba MCP acima também não
substitui nenhuma dessas integrações; clientes como OpenCode podem manter sua
própria configuração MCP.

#### Aba AMD GPU

Telemetria AMD via sysfs, com **auto-refresh de 2 s**. O toggle permite desligar
esse ciclo, e o botão **atualizar** faz uma leitura manual ([AmdPage.tsx](app/src/components/AmdPage.tsx#L82-L203)).
Há um card por GPU AMD válida, além dos totais agregados de VRAM; cada leitura de
sensor é opcional: quando o sysfs não fornece temperatura, utilização, fan,
energia ou clocks, a UI mostra `N/A`, sem invalidar os outros cards
([amd.py](app/api/core/amd.py#L193-L214) e [L253-L320](app/api/core/amd.py#L253-L320)).
Os cards expõem VRAM total/usada/livre, temperatura edge/memória/hotspot e
limites, utilização, fan, power, clocks e `driver_version` por placa
([amd.py](app/api/core/amd.py#L270-L295) e [AmdPage.tsx](app/src/components/AmdPage.tsx#L375-L423)).
O resumo agregado também expõe `host_temp_c`; sensores ausentes retornam `N/A`
por card e `host_temp_c` pode ser `null`, sendo mostrado como `N/A` na UI
([amd.py](app/api/core/amd.py#L171-L189), [L298-L320](app/api/core/amd.py#L298-L320) e [AmdPage.tsx](app/src/components/AmdPage.tsx#L238-L250)).
Os gráficos de histórico de VRAM e RAM são mantidos somente em memória no
frontend, com no máximo **60 pontos**; não há série histórica persistida no
backend ([AmdPage.tsx](app/src/components/AmdPage.tsx#L16-L17) e [L130-L155](app/src/components/AmdPage.tsx#L130-L155)).
A execução E2E mais recente marcou `GPU-01` a `GPU-07` como PASS
(evidência registrada no repositório privado (logs/final/6.log)).

#### Aba Settings

![Tela Settings](telas/tela%20de%20configurações.png)

Onde os caminhos são configurados (ver [seção 2.4](#24-configurar-os-caminhos)):

- **Pastas de modelos** — adicione/remova quantas quiser; a aba Models revarre ao salvar. A primeira pasta é a sugerida como destino de download.
- **Diretórios dos binários llama.cpp** — um campo por backend (turbo, vanilla e custom; o mtp herda do vanilla). Vazio = usa o default do código; o texto `usando default: …` mostra qual é. O botão ↺ restaura o default. Informe a **pasta** que contém `llama-server`/`llama-cli`, não o executável.
- O campo **custom** é o único cujo default não existe de fábrica: é onde você aponta o build específico de um modelo. Enquanto estiver vazio, o backend `custom` aparece no editor como *"sem caminho — configure em Settings"* e não pode ser lançado. O CLI lê o mesmo arquivo (menu **settings → 🧰 Diretórios dos binários llama.cpp**), então configurar num lado vale nos dois.

---

## 4. O CLI (models.py)

Tudo que a APP faz de essencial, em modo texto interativo (setas ↑↓ + Enter):

```bash
cd /path/to/llm-launcher-amd
python3 models.py
```

### Menu inicial

```
⚙️  Modo de execução:
❯ llama-server  — API HTTP (OpenAI-compatible, Claude Code, etc.)
  llama-cli     — Chat interativo no terminal
  download      — Baixar modelo do HuggingFace
  delete        — Apagar um modelo (.gguf) e o mmproj associado do disco
  settings      — Configurar pastas onde procurar modelos (.gguf)
```

### Modo `llama-server` (o principal)

1. **Backend** — turbo / vanilla / mtp / custom / dflash (só aparecem os que têm binário no caminho configurado — inclusive o override de pasta feito em Settings; o CLI ainda suporta o backend experimental `dflash`, de decodificação especulativa com drafter GGUF separado, que não existe na APP).
2. **Modelo** — lista tudo que foi achado nas pastas configuradas, marcando modelos com visão (🖼️).
3. **Configuração salva?** — se você já rodou esse par modelo+backend antes, o CLI mostra a config anterior num quadro e pergunta se quer reutilizá-la. Respondendo **sim**, pula direto para o launch. As configs são as **mesmas da APP** (mesmo arquivo).
4. Caso contrário, a primeira pergunta é **"deixar o llama.cpp decidir tudo?"** — respondendo **sim**, o CLI só pergunta mmproj e verbose e sobe com o comando mínimo (equivale ao checkbox da APP; nada de tuning entra na linha de comando e o auto-degrade fica desligado).
5. Não sendo auto, pergunta passo a passo: **context window**, **KV cache** (o CLI *sonda o binário* e só oferece os tipos que aquele build aceita), **flash attention**, **`-ngl`** (com sugestão automática baseada na VRAM livre e menu de estimativa por opção), **`-ncmoe`** para MoE, **split entre GPUs** quando há mais de uma, **slots paralelos**, **cache-ram**, **ctx-checkpoints**, **reasoning budget** (para modelos thinking), **max-tokens**, **batch/ubatch**, **mlock**, **verbose**.
6. Antes de subir, imprime a **estimativa de memória** (as mesmas contas do painel da APP, com barras no terminal) e o **comando completo**. No modo auto a estimativa é omitida — quem escolhe ctx/`-ngl`/KV ali é o llama.cpp, então qualquer número nosso seria chute.
7. Executa com **launch resiliente**: monitora a saída, classifica falhas (OOM, KV rejeitado, flag desconhecida, GGUF corrompido…) e aplica o [auto-degrade](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai) até o servidor estabilizar. A config que funcionou é salva para a próxima vez.

### Modo `llama-cli`

Mesmo fluxo de seleção, mas gera um chat interativo direto no terminal em vez do servidor HTTP.

### Modos utilitários

- **download** — igual à aba Download da APP: cola URL/id do HF ou busca por texto, escolhe a quantização, baixa com barra de progresso e validação.
- **delete** — apaga um `.gguf` (e mmproj associado) com confirmação.
- **settings** — adiciona/edita/remove as pastas de modelos (grava no mesmo `app_settings.json` da APP).
- **status da GPU** — telemetria disponível no host; a APP expõe a aba **AMD GPU**.

---

## 5. Onde ficam os arquivos de configuração

Todos são JSON simples, compartilhados entre APP e CLI:

| Arquivo | Conteúdo |
|---|---|
| `app_settings.json` | pastas de modelos (`model_paths`) e diretórios dos binários (`backend_paths`) |
| `last_config.json` | as configurações de launch salvas (o grid da aba Configs / as "configs salvas" do CLI) |
| `fail_history.jsonl` | histórico de falhas de launch e do que o auto-degrade fez (um JSON por linha, para diagnóstico) |

Quando usado, `mcp_servers_config` é sempre o caminho absoluto canônico para
`config/mcp/servers.json`; esse arquivo real fica fora do versionamento.

Por padrão eles moram na raiz deste fork (`last_config.json`,
`fail_history.jsonl` e `app_settings.json`). O local é definido por
`CONFIG_FILE`/`FAIL_HISTORY_FILE` em `app/api/core/constants.py` e em
`models.py`; esses caminhos são absolutos e derivados da raiz do repositório.

---

## 6. Auto-degrade: o que acontece quando o servidor cai

Subir um modelo grande é tentativa e erro: a estimativa pode dizer que cabe e a VRAM real dizer que não. Em vez de simplesmente falhar, o runner (APP e CLI usam a mesma lógica):

1. Lê o log do `llama-server` e **classifica a falha**: OOM de VRAM, tipo de KV não suportado pelo build, flag desconhecida, erro de mmproj, GGUF corrompido/incompleto, crash genérico.
2. Aplica **um passo de degrade** adequado à falha — por exemplo: reduzir `-ngl` (menos camadas na GPU), subir `-ncmoe` (mais experts na CPU), trocar o KV cache por um tier mais comprimido, remover a flag não suportada.
3. Tenta de novo, e repete até estabilizar ou esgotar as opções.

Regras importantes:

- **A janela de contexto nunca é cortada** — o degrade sacrifica velocidade (mais coisas na CPU), não a capacidade que você pediu.
- **GGUF corrompido não entra na escada** — é erro de download (shard faltando etc.); o runner desiste na hora e avisa, porque degradar não conserta arquivo quebrado.
- **Config com "llama.cpp decide" também não** — nenhuma flag nossa vai no comando, então todo degrau geraria a mesma linha e o restart viraria loop; o runner para na primeira queda e mostra o erro.
- Cada tentativa aparece numerada no modal de logs da APP (ou no terminal, no CLI), e a config final que funcionou é a que fica salva.

---

## 7. Solução de problemas

**A APP web mostra "❌ conectando ao backend Python…" / erro de conexão**
O backend FastAPI não subiu. Confira se `app/.venv` existe, rode `npm start`
em `app/` e confirme a porta escolhida no stdout ou em `logs/STATUS.json`.

**Badge do backend vermelho / "Nenhum binário do llama.cpp encontrado"**
O `llama-server` não está no caminho esperado. Confira se a compilação (seção
2.2) terminou sem erro e se o diretório na aba **Settings** aponta para a
pasta `build/bin` correta.

**A lista de modelos está vazia**
Cadastre em **Settings → Pastas de modelos** a pasta onde estão seus `.gguf` (ou baixe um pela aba Download). O scanner é recursivo.

**O download exige "pasta cadastrada"**
Por segurança, só se baixa para pastas explicitamente cadastradas no Settings. Adicione a pasta primeiro.

**O modelo carrega e cai na hora (out of memory)**
Deixe o auto-degrade trabalhar — ele reduz `-ngl`/aumenta `-ncmoe` sozinho. Para evitar de vez: use o botão **sugerir -ngl** no editor, um KV cache quantizado (`q8_0`), ou uma quantização menor do modelo.

**Erro "data is not within the file bounds" / "invalid magic number"**
GGUF corrompido ou download incompleto (típico de modelo multi-shard interrompido). Apague pela aba Models e baixe de novo.

**Tokens por segundo muito baixos com a GPU ociosa**
Veja a aba **AMD GPU** e a VRAM livre no host. Também confira se `-ngl` não caiu
demais após um auto-degrade antigo salvo na config.

**Duas GPUs, desempenho pior que uma**
Use split mode **layer** (default), não **row** — no editor, campo "Split entre GPUs (-sm)".

---

## 8. Segurança e advisories conhecidos

Os defaults são conservadores: o backend web e o `llama-server` ficam em
loopback (`127.0.0.1`), e o MCP fica desligado. O `POST /api/settings` só é
aceito em loopback (uma tentativa remota recebe **403**). Para habilitar MCP,
use `LLM_LAUNCHER_ENABLE_MCP=1`; ele continua loopback-only. O override
`LLM_LAUNCHER_ALLOW_REMOTE_MCP=1` é deliberadamente perigoso: expõe um executor
de comandos e pode permitir RCE no ambiente, portanto só use em uma rede e
processo de confiança.

Paths recebidos pela API são contidos nas raízes cadastradas: modelos, mmproj,
deleções, downloads e sidecars rejeitam traversal, escapes por symlink e
destinos não cadastrados. Downloads usam `.part`, validam tamanho quando a
origem informa o tamanho e só substituem o arquivo final no término; `origin.json`
e `sampling.json` são sidecars best-effort igualmente contidos. O destino de
download deve ser uma pasta de modelos já cadastrada em Settings.

Advisories conhecidos desta fase:

- `npm audit` reporta **5** advisories (**3 high, 1 moderate, 1 low**),
  medidos com Node 22 via nvm; não foram corrigidos.
- A auditoria das dependências Python reportou **0** advisories.
- Trivy e OSV não foram executados nesta fase.
- A verificação histórica do Gitleaks ficou limpa.
- `models.py` na raiz e os scripts `bench_*.py` são ferramentas locais, não
  fazem parte do runtime web. Eles mantêm shell/`exec` intencional para uso
  local; não devem receber nem ser expostos a inputs não confiáveis.
- `--mcp-servers-config` também inicia comandos locais definidos no JSON; use
  `command` absoluto e somente arquivos/binários confiáveis. A aba MCP usa
  `shell=True` por desenho e é outro executor local, sem implementar MCP.

A APP web é headless e usa a telemetria AMD disponível no host. As releases são
geradas pelo workflow do GitHub Actions após uma tag `v*`; consulte o histórico
de releases e reporte problemas de segurança de forma responsável ao mantenedor
antes de publicar detalhes exploráveis.

## 9. Licença

Este projeto é distribuído sob a licença **[PolyForm Noncommercial 1.0.0](LICENSE)**. Em resumo:

- ✅ **Pode**: clonar, usar, estudar, modificar e redistribuir (inclusive com suas alterações) para qualquer fim **não comercial** — uso pessoal, estudo, pesquisa, hobby, instituições de ensino e organizações sem fins lucrativos;
- ❌ **Não pode**: vender o software, cobrar por ele ou usá-lo como parte de um produto/serviço comercial;
- 📄 Redistribuições devem manter uma cópia da licença (ou o link para ela).

O texto completo (que é o que vale juridicamente) está no arquivo [`LICENSE`](LICENSE).
