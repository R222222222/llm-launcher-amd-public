# ADR 0003 — Loopback por padrão e Tailscale opt-in

## Status

Aceita, corrigida e implementada.

## Contexto

O registro histórico inicial descrevia bind obrigatório em um endereço
Tailscale, e o host de validação foi de fato iniciado em `<tailscale-ip>`.
Essa era uma premissa de implantação daquela execução, não um requisito
intrínseco do launcher. Exigir uma interface Tailscale impediria uso local e
faria o default depender de uma rede externa.

## Decisão

O backend FastAPI usa `127.0.0.1` quando `LLM_LAUNCHER_HOST` não está definido.
O `llama-server` usa `127.0.0.1` quando `LLM_LAUNCHER_LLAMA_HOST` não está
definido. A publicação em Tailscale é voluntária: o operador define
explicitamente o(s) endereço(s) confiável(is). Os dois overrides são
independentes.

Em desenvolvimento, CORS só é habilitado com
`LLM_LAUNCHER_DEV_CORS=1` e fica limitado a `localhost:5173` e
`127.0.0.1:5173`. Escrita de Settings permanece protegida por loopback.

## Consequências

- Uma instalação nova não escuta na rede por acidente.
- Tailscale continua suportado como caminho de publicação opt-in, não como
  bind obrigatório.
- Evidências históricas com `<tailscale-ip>:8420` devem ser lidas como uma
  configuração explícita de execução, não como o default atual.
- Alterar somente o host da API não altera automaticamente o host do
  `llama-server`; ambos precisam ser configurados quando a publicação externa
  for desejada.
- O modelo de ameaça fica mais simples no uso local, mas publicar a aplicação
  continua exigindo autenticação/rede confiável fora do escopo desta ADR.
