# ADR 0001 — Remover Electron

## Status

Aceita e implementada.

## Contexto

O fork AMD precisava de uma aplicação web headless, executável em um host Linux
com acesso à GPU e utilizável por um navegador. O Electron adicionaria um
runtime desktop, um processo de empacotamento adicional e uma fronteira que não
é necessária para servir React e expor a API FastAPI. Os logs da Fase 1
registram a remoção do Electron do fork.

## Decisão

Remover Electron do produto web. O frontend é compilado pelo Vite para
`app/dist/` e servido pelo `StaticFiles` do FastAPI depois das rotas `/api`.
Em desenvolvimento, Vite continua disponível como servidor separado com proxy
para o backend.

## Consequências

- O deploy web precisa de Python/FastAPI/uvicorn e Node apenas para instalar e
  compilar o bundle.
- A produção usa uma origem única: o navegador obtém HTML/JS/CSS e API do
  FastAPI.
- Não existe janela Electron nem IPC desktop para documentar ou depurar.
- A UI depende de navegador; o backend e o CLI continuam sendo os pontos
  operacionais sem interface.
- A mudança é uma decisão de produto registrada historicamente na Fase 1; ela
  não implica que toda validação de navegador tenha sido concluída em qualquer
  ambiente.
