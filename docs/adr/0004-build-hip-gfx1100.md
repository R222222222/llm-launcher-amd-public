# ADR 0004 — Build HIP com gfx1100

## Status

Aceita e validada historicamente no host AMD de referência.

## Contexto

O fork precisa de um `llama.cpp` capaz de executar offload na GPU AMD. A Fase 3
registrou a disponibilização do toolchain ROCm/HIP, a compilação do checkout em
`vendor/llama.cpp` e a necessidade de escolher um alvo de GPU compatível com a
Radeon RX 7900 XTX.

## Decisão

Configurar o build com `GGML_HIP=ON` e `GPU_TARGETS=gfx1100`. O checkout usado na
validação histórica foi o upstream em `vendor/llama.cpp`, commit
`6a32c29a746a2e44de463de647f9f6661eb5086b`. A configuração precisou de
`CMAKE_HIP_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11` para que o
clang HIP encontrasse os headers GCC 11 naquele host.

## Consequências

- Os binários esperados ficam em
  `vendor/llama.cpp/build/bin/llama-server` e `llama-cli`.
- A Fase 4 registrou health HTTP 200, delta de VRAM e offload de 29/29 camadas
  na RX 7900 XTX.
- `gfx1100` é a configuração validada para aquele host, não uma promessa de
  portabilidade para todas as GPUs AMD.
- Outras arquiteturas, versões ROCm ou toolchains podem exigir outro alvo e
  outra configuração CMake.
- A construção do fornecedor é separada do código Python/React; Settings e
  `LLM_LAUNCHER_LLAMA_CPP_BIN` permitem apontar para outro diretório de binários.
