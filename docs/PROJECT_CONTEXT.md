# Contexto do projeto

Este é um launcher web headless para modelos GGUF executados por
`llama-server`, com backend FastAPI e frontend React/Vite. A especificação
autoritativa é `SPEC_FINAL.md`; decisões e resultados
devem ser interpretados à luz dela e da evidência registrada nos logs.

## Estado atual

- A branch é `feature/amd-web`.
- Os Gates 1–7 estão concluídos com `PASS`.
- A Fase 7 consolidou a documentação e foi encerrada após revisão factual
  independente; a Fase 8 (publicação) é a próxima fase operacional.
- O último commit da fase concluída é o HEAD da Fase 6,
  `f6002886785f193c97ab2a00241e410f348d1822`. Ele é a referência da base
  concluída; commits posteriores de documentação podem mover o HEAD do clone.
- A evidência final da Fase 6 registra 38 `PASS`, 9 `FAIL` e 26 `NÃO
  VERIFICADO` na evidência registrada no repositório privado (logs/final/6.log). Os números são
  contagens daquela execução, não uma conversão dos itens não verificados em
  aprovação.

O hardware validado é uma Radeon RX 7900 XTX em configuração de uma GPU.
Não há validação de execução multi-GPU, placement de camadas/tensores ou
split entre placas; testes de contrato e enumeração não constituem essa
validação.

## Telemetria AMD atual

A fronteira AMD enumera placas DRM/sysfs AMD elegíveis e agrega VRAM a partir
de `mem_info_vram_total` e `mem_info_vram_used`. Por placa, o contrato atual
expõe estes campos:

- Memória: `memory.total`, `memory.used`, `memory.free`.
- Temperatura: `temperature.gpu`, `temperature.memory`,
  `temperature.hotspot`, `temperature.gpu.limit` e
  `temperature.gpu.tlimit`.
- Utilização e energia: `utilization.gpu`, `utilization.memory`, `power.draw`
  e `power.limit`.
- Ventoinha e clocks: `fan.speed`, `clocks.sm` e `clocks.mem`.
- Driver: `driver_version`.

As fontes são os arquivos sysfs DRM, os diretórios hwmon associados e
`/sys/module/amdgpu/version` (com fallback para a versão do kernel). Cada
leitura de sensor é tratada de forma independente: sensor ausente, inválido ou
não exposto afeta seu campo e os campos derivados dele (por exemplo, `edge` ou
o limite ausente também deixa `temperature.gpu.tlimit` indisponível), sem
inventar valor nem invalidar campos não dependentes. `host_temp_c` vem de
`psutil.sensors_temperatures()` quando houver leitura disponível; ausência não
deve ser tratada como uma medição. Não há alegações de valores atuais para
sensores que não tenham sido lidos.

O build AMD validado historicamente usou `GGML_HIP=ON`,
`GPU_TARGETS=gfx1100` e ROCm 6.3 no host com a RX 7900 XTX. Isso não é uma
matriz de compatibilidade nem prova de suporte multi-GPU para outras placas.

## Fronteira da UI

A UI gerencia configurações, modelos, launch, logs, download e stop; ela não é
uma tela de chat. A validação histórica de `/v1/chat/completions` foi feita
diretamente contra o `llama-server`, pela API compatível com OpenAI, e não por
um chat dentro da aplicação. Não atribuir à UI capacidades de conversa que
pertencem ao servidor/API.

## Quickstart real

Os comandos abaixo pressupõem Linux, Python 3.10+, Node.js 18+ e
CMake/toolchain C++. Neste host, Node.js 18+ deve ser carregado pelo nvm em um
shell de login ou com `source ~/.nvm/nvm.sh && nvm use 22`.

```bash
git clone <url-do-fork> llm-launcher-amd
cd llm-launcher-amd

python3 -m pip install --user virtualenv
python3 -m virtualenv app/.venv
app/.venv/bin/python -m pip install -r app/requirements.txt
app/.venv/bin/python -m pip install -r app/requirements-dev.txt

mkdir -p vendor
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
git -C vendor/llama.cpp checkout 6a32c29a746a2e44de463de647f9f6661eb5086b
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build -j

cd app
npm ci
npm run build
npm start
```

O comando pressupõe um toolchain ROCm/HIP funcional. No host de referência, o
clang HIP também precisou de
`-DCMAKE_HIP_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`; essa
correção é específica do host e não deve ser generalizada.

Por padrão, o FastAPI escuta em loopback, escolhe uma porta livre a partir de
8420 e reserva 8421 para o `llama-server`. O bundle em `app/dist/` é servido
pelo próprio FastAPI; em produção o frontend usa same-origin.

Para desenvolvimento separado, mantenha `npm start` em um terminal e rode
`npm run dev` em outro dentro de `app/`. O proxy Vite encaminha `/api` para
`http://127.0.0.1:8420` por padrão; `VITE_DEV_API_TARGET` pode apontar para
outro backend de desenvolvimento.

O launcher procura modelos em `runtime/models` e binários em
`vendor/llama.cpp/build/bin`. Os caminhos podem ser sobrescritos por
`LLM_LAUNCHER_MODELS_DIR` e `LLM_LAUNCHER_LLAMA_CPP_BIN`, ou configurados em
**Settings**. O scanner procura `.gguf` recursivamente.

## Limites operacionais

1. A telemetria depende dos arquivos DRM/sysfs disponíveis no host; ela não
   depende de `rocm-smi`.
2. A VRAM agregada é uma visão contábil das placas AMD elegíveis e não prova
   distribuição de pesos entre elas.
3. O runner usa grupo de processos isolado, `shell=False`, preflight da porta
   e readiness por `/health`; o cancelamento é best-effort.
4. O backend web e o `llama-server` ficam em loopback por padrão. MCP é
   desabilitado por padrão; habilitá-lo cria uma fronteira explícita de
   execução de comandos e exige revisão de segurança.
5. A estimativa de VRAM/RAM é heurística; o runtime pode exigir auto-degrade
   ou falhar por motivos que ela não modela.

## Mapa de leitura

1. **Este arquivo** — estado, limites e quickstart.
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — componentes, execução, isolamento e
   telemetria.
3. `SPEC_FINAL.md` — requisitos e gates autoritativos.
4. ADRs em [`adr/`](adr/) — decisões registradas.

### Fontes primárias do clone

- Visão operacional e instalação: [`README.md`](../README.md).
- API e bind: [`app/api/server.py`](../app/api/server.py).
- AMD, execução e processo: [`amd.py`](../app/api/core/amd.py),
  [`runner.py`](../app/api/core/runner.py) e
  [`running.py`](../app/api/core/running.py).
- Frontend e contrato HTTP: [`App.tsx`](../app/src/App.tsx),
  [`client.ts`](../app/src/api/client.ts) e
  [`AmdPage.tsx`](../app/src/components/AmdPage.tsx).
- Estado da fase: evidência registrada no repositório privado (logs/STATUS.json) e
  evidência registrada no repositório privado (logs/final/7.log). A execução E2E permanece na
  evidência registrada no repositório privado (logs/final/6.log).
