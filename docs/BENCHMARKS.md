# Benchmarks

Resultados do smoke test de produção da Fase 8, executado no fork AMD com
`llama.cpp`/ROCm e uma GPU AMD. O resultado global foi `PASSED_WITH_NOTES`.

| Modelo/variante | Backend e configuração | Pico de VRAM | Resultado de desempenho |
|---|---|---:|---:|
| Ornith | `vanilla`, contexto `16.384`, KV `f16`, `-ngl 99`, `-np 1` | **20.870 MiB** | **110.215 tok/s** (coding anchor) |
| Fable regular, non-MTP | `vanilla`, contexto `32.768`, KV `f16`, `-ngl 99`, `-np 1` | **19.298 MiB** | **32.267674 tok/s** |
| Fable NEO-MTP | `mtp`, contexto `32.768`, KV `f16`, `-ngl 99`, `-np 1`, `draft-mtp`, `max 2` | **20.288 MiB** | **44.240789 tok/s**; aceitação **323/380 = 85%** |

## Comparação observada

No protocolo usado, Fable MTP mediu `+11,973115 tok/s`, ou **+37,11%**,
em relação ao Fable non-MTP. Esse número é uma observação deste protocolo,
não um speedup geral ou causal.

As referências do Ollama — aproximadamente `97,5 tok/s` para Ornith e
`31,4 tok/s` para Fable — são **descritivas e não controladas**. Os engines,
backends, modelos/variantes e configurações não foram igualados; portanto não
servem como comparação experimental de desempenho nem como garantia de ganho.

## Fontes primárias

- STATUS da execução — evidência registrada no repositório privado (logs/STATUS.json), incluindo hashes, configurações,
  picos de VRAM e métricas de aceitação.
- Log completo da Fase 8 — evidência registrada no repositório privado (logs/fase8.log), com os launches, timings,
  respostas e encerramento pela UI.
- Relatório final — evidência registrada no repositório privado (logs/MORNING_REPORT.md), com a adjudicação e os caveats.
- Respostas e métricas detalhadas em
  evidência registrada no repositório privado (logs/fase8_responses/).

---

## Fase 3 — perfis seed (E2E diário com `--profile-e2e`)

Execução diária dos 5 perfis seed (`docs/profiles/seed-profiles.json`), cada um
com: estimativa de VRAM com folga positiva → launch → `/health` 200 →
completion real (texto; imagem PNG determinística nos perfis de visão) →
cancel → baseline de VRAM restaurado e zero processos órfãos. Tokens/s =
`completion_tokens / tempo decorrido` (inclui raciocínio dos modelos thinking).

| Perfil | Modelo | Backend | Contexto | KV | Folga estimada | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| `agente-codigo-27b-mtp` | Qwen3.6-27B-Fable-NEO-MTP `Q4_K_M` + MTP | `mtp` | 65.536 | `q8_0` | +1.514 MiB | **44,406** |
| `contexto-longo-27b` | Qwen3.6-27B-Fable `Q4_K_M` | `vanilla` | 98.304 | `q8_0` | +1.697 MiB | **30,607** |
| `chat-ferramentas-ornith` | Ornith-1.0-35B `Q4_K_M` | `vanilla` | 65.536 | `q8_0` | +1.771 MiB | **101,649** |
| `visao-27b-mtp` | Qwen3.6-27B-Fable-NEO-MTP `Q4_K_M` + `mmproj-F16.gguf` | `mtp` | 32.768 | `q8_0` | +1.827 MiB | **37,957** |
| `ocr-glm` | GLM-OCR `f16` + `mmproj-GLM-OCR-Q8_0.gguf` | `vanilla` | 131.072 | `q8_0` | +16.820 MiB | **51,182** |

Observações:

- `gpu_layers` 99, `flash_attn` e `sampler_source` do seed em todos os perfis;
  o `ocr-glm` usa amostragem manual (temp 0.1, top_k 1) — os demais seguem o
  template (que resolve temp 0.3/top_p 0.8 via `/api/models/sampling`).
- Perfis de visão: completion com imagem PNG e prompt `OCR`; resposta esperada
  `Hello OCR 12345` encontrada, e `loaded multimodal model` presente no log do
  launch. O `ocr-glm` respondeu em 17 tokens (0,332 s).
- Nenhum perfil degradou (`degrade_events` vazios), nenhum `llama-server`
  órfão após o Stop, e a VRAM volta ao baseline de 26 MiB em todos os casos.
- Evidência por perfil em `logs/diario/<perfil>.json` e
  `logs/diario/profiles-e2e-summary.json`.

### Fórmula de VRAM (Q4_K_M na GPU, KV `q8_0`)

A estimativa de VRAM usada pelo seed e pela E2E é, em MiB:

```
VRAM ≈ 20091 + (mmproj ? 1151 : 0) + 45,75 × (ctx / 1024)
```

onde o termo de KV cresce com o contexto e o termo de mmproj entra apenas nos
perfis de visão. A GPU disponível mede 24.533 MiB livres com o backend parado.
Com KV `f16` (o dobro), o 27B não cabe acima de 32.768 de contexto — a 65.536
a estimativa fica negativa (≈ −646 MiB), confirmando a necessidade de `q8_0`
nos perfis grandes.
