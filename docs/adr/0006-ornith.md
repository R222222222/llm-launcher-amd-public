# ADR 0006 — Perfil de validação do Ornith

## Status

Aceita como perfil histórico de validação; não é default universal.

## Contexto

A Fase 8 precisava exercitar um modelo grande com o backend upstream vanilla e
registrar uma configuração reproduzível. O blob Ornith usado foi identificado
por SHA-256 e testado no host AMD de referência.

## Decisão

Para o cenário Ornith documentado, usar:

| Campo | Valor |
|---|---|
| Backend | `vanilla` |
| Contexto | `16384` |
| KV K/V | `f16` |
| Camadas GPU | `99` |
| Slots | `1` |

O builder emite o pool KV unificado quando o binário suporta `--kv-unified`,
além das flags normais de servidor. A identidade do blob, o tamanho, o pico de
VRAM e os resultados estão nos logs da Fase 8; esta ADR não promove essas
medições a benchmark geral.

## Consequências

- A combinação passou na primeira tentativa de carregamento no protocolo
  registrado, com pico de 20.870 MiB.
- Coding, tool-call e multi-turn foram adjudicados como `PASS`; Think foi
  `PASS_WITH_NOTE`.
- O anchor registrado foi `110.215 tok/s`, comparado apenas descritivamente
  dentro daquele protocolo; não é garantia de throughput em outro host,
  prompt, versão ou configuração.
- `f16`, contexto 16k e `ngl=99` são uma receita de validação, não uma regra
  para toda máquina ou todo modelo Ornith.
