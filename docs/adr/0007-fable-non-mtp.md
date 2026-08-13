# ADR 0007 — Perfil Fable non-MTP

## Status

Aceita como perfil histórico de validação; não é uma declaração sobre o blob
MTP.

## Contexto

Era necessário comparar uma variante Fable regular sem ativar decodificação
especulativa. O blob local usado no teste foi a variante identificada por
`8440...`, distinta do blob MTP `c796...`. O nome local e essa distinção devem
ser preservados: o relatório não autoriza reclassificar o regular como MTP ou
como o blob oficial MTP.

## Decisão

Usar o backend `vanilla`, contexto `32768`, KV K/V `f16`, `ngl=99` e `np=1`.
Não emitir `--spec-type` nem `--spec-draft-n-max`. A ausência dessas flags é
parte do contrato do perfil non-MTP.

## Consequências

- O log histórico confirmou `-c 32768`, `-ngl 99`, `f16`, `-np 1` e ausência de
  `spec-type`.
- Logic foi `PASS` e Multi foi adjudicado `PASS_WITH_NOTE`; os resultados raw
  mais estritos permanecem preservados.
- O anchor registrado foi `32.267674 tok/s`. É uma observação daquele protocolo,
  não um benchmark generalizável nem comparação causal com Ollama.
- A configuração mantém o caminho vanilla normal e deixa o perfil MTP separado,
  evitando que uma flag especulativa seja aplicada a um modelo não MTP.
