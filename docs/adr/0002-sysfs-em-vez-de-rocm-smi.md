# ADR 0002 — Usar sysfs em vez de rocm-smi

## Status

Aceita e implementada.

## Nota de supersessão

A limitação histórica que declarava temperatura e utilização indisponíveis foi
superada pela [ADR 0009 — Paridade de telemetria AMD via sysfs](0009-paridade-de-telemetria-amd-via-sysfs.md).
Sensores continuam opcionais e retornam `N/A`/`null` quando o host não os
expõe; a decisão desta ADR sobre sysfs permanece a base da implementação.

## Contexto

O launcher precisa mostrar VRAM AMD sem depender de um executável externo, de
uma versão específica das ferramentas ROCm ou de parsing de saída textual.
`rocm-smi` é útil para diagnóstico do host, mas não é uma base estável do
contrato HTTP da aplicação. O histórico mostra seu uso pontual em verificações
de processos KFD; isso não é a implementação da telemetria.

## Decisão

Implementar a telemetria em `app/api/core/amd.py` lendo diretamente
`/sys/class/drm/card*/device/vendor`, `mem_info_vram_total` e, quando presente,
`mem_info_vram_used`. Aceitar somente vendor AMD `0x1002`, enumerar novamente
em cada chamada e agregar os cards válidos.

## Consequências

- Não há dependência de `rocm-smi` para iniciar o backend ou responder
  `/api/gpu`.
- Um card sysfs defeituoso é isolado dos demais.
- A resposta pode preservar o total e declarar uso/livre indisponíveis se um
  card não fornecer `mem_info_vram_used`.
- A soma de VRAM representa contabilidade agregada; não valida split, afinidade
  HIP ou desempenho multi-GPU.
- A limitação de temperatura e utilização acima é histórica e está supersedida
  pela [ADR 0009](0009-paridade-de-telemetria-amd-via-sysfs.md); a
  disponibilidade continua dependente dos sensores do host.
- A implementação é preparada para múltiplos cards, mas o hardware real
  validado tinha apenas uma RX 7900 XTX. Multi-GPU foi coberto somente por
  testes unitários/contratuais.
