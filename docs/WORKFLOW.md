# Workflow único: do zero ao play

Este é o caminho operacional completo para este fork: preparar o host, compilar,
subir a APP, semear os perfis, lançar um modelo, verificar a API, usar a WebUI e
encerrar sem deixar processos. Os defaults são deliberadamente locais; a regra do
projeto é loopback com túnel SSH, não exposição na rede (Regras atuais).

## Uso diário

Sentar e usar, sem preparo manual — o resto deste documento detalha cada passo.

1. **No host do backend:** o backend sobe automaticamente no boot via
   systemd user unit (`scripts/systemd/llm-launcher.service`, habilitada com
   `systemctl --user enable`). Para iniciar manualmente:
   `scripts/start-backend.sh` (idempotente; sobe o backend em loopback e
   loga em `logs/backend.log`). Para instalar a unit:
   `cp scripts/systemd/llm-launcher.service ~/.config/systemd/user/ &&
   systemctl --user daemon-reload && systemctl --user enable llm-launcher`.
   A unit usa `%h` (home do usuário via systemd specifier) — sem caminho
   hardcoded.
2. **No Windows:** `scripts/tunnel.ps1 -Hostname <ip>` (ou defina
   `LLM_LAUNCHER_HOST`); abre o túnel SSH das portas 8420/8421.
3. **Abra** `http://127.0.0.1:8420` no navegador.
4. **Escolha o perfil** na aba Configs e clique em **launch** — play.
5. **(Desenvolvedores)** Para rodar a suíte E2E diária,
   `scripts/setup-e2e.sh` cria/repara o ambiente em
   `~/.cache/llm-launcher-amd/e2e` (virtualenv + `patchright==1.61.2`), e o
   cenário com todos os perfis roda com
   `~/.cache/llm-launcher-amd/e2e/bin/python tests/e2e/run.py --profile-e2e`
   ([tests/e2e/README.md](../tests/e2e/README.md)).

Detalhes: [loopback e túnel](#loopback-e-túnel), [subir a APP e semear os
perfis](#subir-a-app-e-semear-os-perfis), [launch, health, completion e
Stop](#launch-health-completion-e-stop) e [WebUI, router e MCP](../README.md#L396-L454).

## Pré-requisitos e build

Instale Git, Python **3.10 ou posterior**, Node.js **18 ou posterior**, CMake
**3.24 ou posterior** e um compilador C/C++ ([requisitos do README](../README.md#L59-L78)).
Depois:

```bash
git clone <fork-url> llm-launcher-amd
cd llm-launcher-amd

mkdir -p vendor
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp
git -C vendor/llama.cpp checkout 6a32c29a746a2e44de463de647f9f6661eb5086b
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/llama.cpp/build -j

python3 -m pip install --user virtualenv
python3 -m virtualenv app/.venv
app/.venv/bin/python -m pip install -r app/requirements.txt
app/.venv/bin/python -m pip install -r app/requirements-dev.txt
```

O layout acima produz `vendor/llama.cpp/build/bin/llama-server` e
`llama-cli`; o commit e as flags AMD/HIP são os do build validado
(configure corrigido e build/versionamento/resultados — evidência registrada no repositório privado (logs/fase3.log)). Em um host de referência cujo
clang HIP não encontre os headers GCC, pode-se acrescentar
`-DCMAKE_HIP_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11`; essa é
uma correção específica daquele host, não um requisito geral
([BUILD_LOG](BUILD_LOG.md#L46-L57)). Os defaults e a instalação estão também no
[README](../README.md#L93-L120).

Se os modelos ou binários estiverem em outro lugar, configure-os na aba Settings
ou use os overrides documentados no [README](../README.md#24-configurar-os-caminhos).
O seed abaixo exige que os GGUF referenciados pelo manifesto já existam.

## Loopback e túnel

O backend faz bind loopback na primeira porta livre a partir de `8420`, mas
pula `8421`, reservada ao `llama-server`; portanto `8420` é o default quando
está livre, não uma garantia se outra porta já estiver ocupada
([server.py](../app/api/server.py#L1160-L1181)). O `llama-server` usa
`127.0.0.1:8421` por padrão ([constants.py](../app/api/core/constants.py#L31-L44));
ambos os hosts podem ser substituídos, separadamente, por
`LLM_LAUNCHER_HOST` e `LLM_LAUNCHER_LLAMA_HOST` ([server.py](../app/api/server.py#L90-L99)).

Quando o backend roda em outra máquina, abra no cliente o encaminhamento
canônico dos dois serviços:

```bash
ssh -N -L 8420:127.0.0.1:8420 -L 8421:127.0.0.1:8421 usuario@host
```

Assim, o navegador e os clientes locais continuam usando loopback. Não troque os
dois binds por `0.0.0.0`; o acesso remoto previsto é o túnel
(Regras atuais).

## Subir a APP e semear os perfis

Em um terminal no host do backend:

```bash
cd app
npm ci
npm run build
npm start
```

`npm start` usa `app/.venv` e inicia o backend; o build e os scripts estão em
[app/package.json](../app/package.json#L5-L11) e
[start-web.mjs](../app/scripts/start-web.mjs#L14-L23).

Antes de iniciar a APP, deixe `8420` livre para que o backend obtenha essa porta.
O backend pode escolher outra porta quando `8420` está ocupado, mas
`scripts/seed-profiles.py` chama fixamente
`127.0.0.1:8420` ([seed-profiles.py](../scripts/seed-profiles.py#L17-L24)). Em
outro terminal, a partir da raiz do repositório, execute o seed versionado:

```bash
app/.venv/bin/python scripts/seed-profiles.py
```

A autoridade dos perfis atuais é
[`docs/profiles/seed-profiles.json`](profiles/seed-profiles.json#L1-L51): ele contém
quatro perfis, incluindo `visao-27b-mtp`. O script lê esse manifesto e valida os
GGUF ([seed-profiles.py](../scripts/seed-profiles.py#L115-L140)), faz
`POST /api/configs` ([seed-profiles.py](../scripts/seed-profiles.py#L17-L31)) e confirma uma ocorrência com campos coincidentes para cada ID
([verificação de persistência](../scripts/seed-profiles.py#L68-L92)). Como a gravação
atualiza pelo `id` ([config_store.py](../app/api/core/config_store.py#L45-L63)), é
seguro executar o comando novamente; as duas execuções bem-sucedidas estão
registradas na evidência registrada no repositório privado (logs/final/5.log).

## Launch, health, completion e Stop

Abra `http://127.0.0.1:8420/` no navegador. Na aba **Configs**, escolha o perfil
`agente-codigo-27b-mtp` e clique em **launch**. Para um teste determinístico, o
alias usado pelo endpoint OpenAI-compatible é o nome do GGUF sem a extensão,
registrado no [harness OpenCode](harness/opencode.md#L49-L61).

Confirme a prontidão do processo e faça uma completion:

```bash
curl -fsS http://127.0.0.1:8421/health

curl -fsS http://127.0.0.1:8421/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M","messages":[{"role":"user","content":"Responda E2E OK"}]}'
```

O caminho crítico validado é launch → `/health` HTTP **200** → completion →
Stop, com a porta de inferência livre e a VRAM retornando ao baseline após o
encerramento (evidência registrada no repositório privado (logs/final/6.log)). Na aba **Configs**
(inclusive no router), use o controle **Stop/parar**. No modal de logs, a ação
correspondente chama-se **Cancelar**; não confunda os dois controles. Depois
confirme a higiene:

```bash
curl -fsS http://127.0.0.1:8420/api/launches
```

O resultado esperado é `[]`; também confirme que `8421` não está mais ouvindo.
Não mate somente a janela do navegador: o backend pode reanexar um
`llama-server` órfão e manter o controle de Stop ([server.py](../app/api/server.py#L781-L807)).

## OCR com GLM-OCR (perfil `ocr-glm`)

O perfil `ocr-glm` usa o GLM-OCR **F16** com `mmproj-GLM-OCR-Q8_0.gguf`,
baixado pela aba Download no commit `65a42de` (SHA-256 e tamanho conferidos,
`origin.json` veraz). Contexto nativo de **131072** cabe folgado: `vram_total`
7713 MiB contra `vram_avail` 24533 MiB (folga 16820 MiB). O alias da API
OpenAI-compatible é o nome do GGUF sem extensão: `GLM-OCR-f16`.

### Sampling fixado manualmente (obrigatório)

O chat template do modelo resolve para `temp 0.6 / top_p 0.95` — preset de
reasoning que **aumenta a alucinação em OCR**. O perfil grava
`sampler_source: "manual"` com `temp: 0.1` e `top_k: 1`, e nada sobrescreve
isso; não troque para o preset do template.

### Formatos de prompt do post oficial

- `"OCR"` — transcrição pura.
- `"OCR markdown"` — estrutura em Markdown (títulos, listas, tabelas).
- `"OCR HTML table"` — tabelas em HTML.
- Variantes equivalentes: `"Text Recognition:"` e `"Table Recognition:"`.

Prompt genérico rende menos. **Imagem primeiro, texto depois** (a ordem usual;
o texto descreve o modo, a imagem é o documento).

```bash
IMG_B64=$(base64 -w0 /tmp/documento.png)
curl -fsS http://127.0.0.1:8421/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"GLM-OCR-f16\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"OCR\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${IMG_B64}\"}}]}]}"
```

Validação real registrada: imagem local determinística (Pillow, texto
`Hello OCR 12345`) devolveu o texto exato com `temp 0.1 / top_k 1`
(evidência registrada no repositório privado (logs/diario/ocr-test.json)).

## WebUI, router e os três limites MCP ([README — MCP](../README.md#L396-L454); [builder](../app/api/core/builder.py#L307-L339); [server](../app/api/server.py#L648-L670); [OpenCode](harness/opencode.md#L81-L99))

### WebUI e router

O frontend servido pelo backend está em `http://127.0.0.1:8420/`; o
`llama-server` lançado fica na API OpenAI-compatible em `127.0.0.1:8421` por
default. Para acesso a partir de outro computador, use os dois forwards SSH
acima. Seleção múltipla na aba Configs usa o router; as configurações escolhidas
são gravadas no preset com `load-on-startup`, e o cliente seleciona o modelo pelo
campo `model` ([server.py](../app/api/server.py#L648-L706)
e [builder.py](../app/api/core/builder.py#L307-L339)). O router não aceita membros
com `mcp_servers_config` ([server.py](../app/api/server.py#L661-L670)).

### MCP do launcher: supervisor de processos

A aba **MCP** do launcher é somente um supervisor de processos locais: recebe
`cwd` e `command`, executa com `shell=True`, captura logs e liga/desliga o
processo. Ela **não** implementa JSON-RPC, `tools/list` nem conecta o processo ao
`llama-server` ([README](../README.md#aba-mcp)). Use esse limite quando quiser
gerenciar um processo auxiliar pela WebUI.

### MCP server-side: `llama-server` + stdio

Para o modelo carregar servidores MCP no startup, crie manualmente
`config/mcp/servers.json` a partir do exemplo versionado. O launcher aceita
somente esse caminho absoluto canônico, valida o schema antes de salvar/build/
launch e então emite condicionalmente `--mcp-servers-config`; o JSON usa
`mcpServers` e um `command` absoluto:

O endpoint `/api/options` usa a chave canônica `mcp_runtime_config` para expor
somente `path`, `exists` e `valid`, sem ecoar comandos, argumentos ou ambiente.

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

Esse é o limite **stdio do próprio llama-server**, diferente da aba MCP e do
MCP do cliente ([builder.py — modo auto](../app/api/core/builder.py#L49-L86),
[builder.py — launch normal](../app/api/core/builder.py#L114-L162) e
SPEC_FINAL). Use somente binários confiáveis;
o arquivo real pode conter comandos sensíveis e deve ficar fora do versionamento.

### MCP client-side: OpenCode

O MCP no `opencode.json` é do cliente OpenCode. Ele é independente do
`mcp_servers_config` e do supervisor da aba MCP; plugins e skills também ficam no
cliente ([harness/opencode.md](harness/opencode.md#L81-L99)). O exemplo atual do
provider local usa a forma de objeto validada para modalidades:

```json
{
  "baseURL": "http://127.0.0.1:8421/v1",
  "models": {
    "ALIAS_DO_MODELO": {
      "attachment": true,
      "modalities": {
        "input": ["text", "image"],
        "output": ["text"]
      }
    }
  }
}
```

Para o arquivo completo, incluindo o provider e o MCP local, use
[`opencode.json.example`](harness/opencode.json.example#L7-L43). Não use a forma
antiga de `modalities` como array simples; a validação atual exige o objeto
`input`/`output` ([opencode.md](harness/opencode.md#L63-L79)). O campo de visão não
prova inferência de imagem: isso depende de um `mmproj` disponível e de um teste
real.

## Estado do E2E mais recente

O registro mais recente é a execução `fase3-profiles-20260813g` (Fase 3, com
`--profile-e2e`): **61 PASS, 0 FAIL e 12 NÃO VERIFICADO**, `rc=0` e cleanup
concluído (checklist — evidência registrada no repositório privado (logs/fase6-e2e/fase3-profiles-20260813g/checklist.json), evidência por perfil — evidência registrada no repositório privado (logs/diario/profiles-e2e-summary.json)). Isso não
significa checklist completamente verde: os 12 itens NÃO VERIFICADO são os que
dependem de um ciclo de download ou de um peer remoto controlado (ex.:
`SETTINGS-06` — veja os itens abertos em [NEXT_STEPS](NEXT_STEPS.md)), e
registros anteriores com falhas adjudicadas permanecem preservados
(evidência registrada no repositório privado (logs/final/6.log)).
