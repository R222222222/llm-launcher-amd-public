# ADR 0008 — Perfil Fable MTP

## Status

Aceita como perfil histórico de validação; `PASSED_WITH_NOTES`.

## Contexto

O launcher expõe `mtp` como um alias do binário `vanilla`, não como um checkout
ou build independente. A variante Fable MTP usada na Fase 8 foi o blob
`c796...`, distinto da variante regular `8440...`. O objetivo era exercitar o
modo MTP nativo disponível no binário vanilla.

## Decisão

Para o perfil Fable MTP, usar:

| Campo | Valor |
|---|---|
| Alias/backend de UI | `mtp` (binário `vanilla`) |
| Contexto | `32768` |
| KV K/V | `f16` |
| Camadas GPU | `99` |
| Slots | `1` |
| Tipo especulativo | `draft-mtp` |
| Máximo de tokens de rascunho | `2` |

O builder só emite `--spec-type draft-mtp` e `--spec-draft-n-max 2` quando o
backend suporta MTP e o modelo é detectado como MTP. O alias `mtp` reutiliza o
binário vanilla por `BACKEND_BINARY_ALIAS`.

## Consequências

- O log de launch confirmou `-c 32768`, `f16`, `-ngl 99`, `-np 1`,
  `--spec-type draft-mtp` e `--spec-draft-n-max 2`.
- Logic foi `PASS`; Multi foi adjudicado `PASS_WITH_NOTE`.
- O protocolo registrou 323 propostas aceitas de 380 (85%) e
  `44.240789 tok/s`, contra `32.267674 tok/s` do cenário non-MTP. O delta de
  37,11% é somente observado nesse protocolo; não é speedup geralizável.
- O consumo maior de VRAM e os resultados dependem do blob, versão do
  llama.cpp, prompts e host. A ADR não transforma a medição em benchmark
  universal.
- Não há uma SPEC de projeto para esse perfil no clone; a decisão é a
  configuração concreta registrada no builder, no frontend e nos logs.
