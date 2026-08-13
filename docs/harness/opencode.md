# Fase 4 — OpenCode como harness

## Resultado

**PASS** nos dois gates da Fase 4:

- `tool_calls` real via MCP cliente, com `duckduckgo_search` concluído;
- quatro chamadas `Task` criadas em background;
- concorrência confirmada pela sobreposição dos subagentes alpha e beta. O
  `max_busy_slots=4` é agregado e inclui requisições primária/title e
  subagentes; não representa quatro subagentes simultâneos.

O PASS segue o gate da SPEC_FINAL: uma tarefa com subagentes e slots
concorrentes observados, não quatro subagentes simultâneos nem uma relação
1:1 entre `id_task` e subagente.

O teste foi feito com OpenCode **1.18.15**, o perfil launcher
`agente-codigo-27b-mtp` e o `llama-server` em
`http://127.0.0.1:8421/v1`. O processo foi parado ao final.

## Isolamento e configuração

A configuração usada foi `/tmp/phase4-opencode/opencode.json`, nunca
`opencode.json` na raiz deste repositório. A execução usou configuração e
diretórios XDG temporários, além de `--pure`:

```bash
OPENCODE_CONFIG=/tmp/phase4-opencode/opencode.json \
OPENCODE_CONFIG_DIR=/tmp/phase4-opencode/config-dir \
XDG_CONFIG_HOME=/tmp/phase4-opencode/xdg \
XDG_DATA_HOME=/tmp/phase4-opencode/data \
XDG_CACHE_HOME=/tmp/phase4-opencode/cache \
XDG_STATE_HOME=/tmp/phase4-opencode/state \
opencode --pure
```

O exemplo versionado em [`opencode.json.example`](opencode.json.example) é
válido, não contém segredo e reproduz essa configuração. O provider habilitado
é somente `llama8421`, usando `@ai-sdk/openai-compatible` e:

```text
baseURL = http://127.0.0.1:8421/v1
model = llama8421/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M
```

O `apiKey` literal `local` é apenas o valor aceito pelo endpoint local; não é
um segredo.

## Perfil, alias e modelo

`agente-codigo-27b-mtp` é o **ID persistente do perfil do launcher**. Ele não é
o nome do modelo no provider. O alias/model key emitido pelo launcher e usado
pelo OpenCode é:

```text
Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M
```

O launcher iniciou esse perfil com `-np 4`, `--spec-type draft-mtp`,
`--spec-draft-n-max 3`, `--metrics`, e confirmou `n_slots = 4`, `load_ok` e
nenhum auto-degrade. O `mmproj` do perfil era `null`.

## Schema de imagem do OpenCode

No OpenCode 1.18.15, o schema atual exige metadados no formato abaixo:

```json
"attachment": true,
"modalities": {
  "input": ["text", "image"],
  "output": ["text"]
}
```

Não usar o formato antigo de array simples. `attachment` e `modalities` são
metadados necessários para o SDK reconhecer a capacidade declarada; **não
comprovam visão** nem processamento de imagem. Como o `mmproj` atual é `null`,
a Fase 4 não declara visão funcional. A Fase 5 decide isso mediante mmproj e
inferência real.

## MCP: cliente, llama-server e aba do launcher

O MCP validado nesta fase é **do cliente OpenCode**. O exemplo configura
`duckduckgo` como servidor local por comando absoluto:

```text
/path/to/.local/bin/uvx duckduckgo-mcp-server
```

Isso é diferente de `mcp_servers_config` / `--mcp-servers-config`, que pertence
ao `llama-server` e configura servidores stdio no startup. Também é diferente
da aba MCP do launcher, que somente supervisiona processos por `cwd` +
`command` com `shell=True`; ela não implementa o protocolo MCP.

O MCP cliente apareceu como `duckduckgo connected`. O evento real foi um
`tool_use` `duckduckgo_search`, com status `completed`, seguido de
`step_finish` com motivo `tool-calls`; a resposta final foi `Config | OpenCode`.

Superpowers é plugin do OpenCode, instalado no cliente. Não entra no launcher.

## Paralelismo

Foram criadas quatro chamadas `Task` em background, todas com
`subagent_type=explore`, arquivos `alpha.txt`, `beta.txt`, `gamma.txt` e
`delta.txt`, e call IDs/session IDs distintos. A concorrência de subagentes é
confirmada diretamente pelos intervalos alpha e beta sobrepostos. A síntese
final foi correta.

O sampler registrou **430 amostras**, `max_busy_slots=4` agregado e
`overlap_samples=354`. Esse agregado inclui requisições primária/title e
subagentes. A primeira sobreposição agregada observada foi nos slots 1 e 3,
com `task88` e `task84`; ela não estabelece mapeamento individual de task para
subagente.

A primeira amostra com quatro slots ativos foi:

```json
{"active":[{"id":0,"id_task":258},{"id":1,"id_task":88},{"id":2,"id_task":259},{"id":3,"id_task":84}],"time_ns":1786408593827909346}
```

`id_task` é o identificador observado pelo sampler; não se infere dele um
mapeamento direto entre cada task e um subagente específico. A amostra é
atividade agregada, não evidência de quatro subagentes simultâneos.

Os intervalos de log que comprovam a sobreposição alpha+beta são:

```text
alpha: 2026-08-11T00:36:21.390Z — 2026-08-11T00:36:41.236Z
beta:  2026-08-11T00:36:30.402Z — 2026-08-11T00:36:43.038Z
overlap: true
```

```text
predicted_tokens_seconds 17.3782
prompt_tokens_seconds 346.563
n_busy_slots_per_decode 2.37255
```

Esses números são observações deste run, sem comparação com 2×3090. O
throughput agregado varia conforme o grau de concorrência.

## Encerramento

O launch `ef9dd83c8bc4` foi cancelado. Após o Stop, `/api/launches` retornou
`[]`, a porta 8421 estava ausente e não havia processo órfão do
`duckduckgo-mcp-server`.

A evidência compactada e reproduzível está em evidência registrada no repositório privado (logs/final/4.log).
