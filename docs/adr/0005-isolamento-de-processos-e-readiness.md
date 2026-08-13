# ADR 0005 — Isolamento PGID/SID, shell=False, preflight e readiness

## Status

Aceita e implementada.

## Contexto

O launcher inicia um processo externo de longa duração e precisa permitir
Stop/restart sem matar o FastAPI, o shell do operador ou processos de outro
serviço. A validação de UI encontrou o risco de considerar o launch pronto
apenas por sinais incompletos e o risco de usar um wrapper shell que pudesse
deixar descendentes vivos.

## Decisão

Para launches POSIX:

- criar o filho com `subprocess.Popen(..., shell=False,
  start_new_session=True)`;
- separar a linha com `shlex.split()` e validar caminhos no builder;
- antes do spawn, recusar a execução se a porta do comando estiver ocupada;
- sondar `/health` sem proxy: HTTP 503 significa carregamento, e readiness
  exige HTTP 200 com JSON `status == "ok"`;
- no cancelamento, sinalizar o grupo somente quando PGID e SID forem o próprio
  PID do launch e o grupo não for o grupo atual; caso contrário, sinalizar só o
  PID;
- persistir PID/configuração em `app/api/api_running.json` e confirmar o nome
  do processo antes de reanexar após restart do backend.

## Consequências

- O Stop histórico encerrou o `llama-server` da própria run e preservou o
  backend web.
- O processo validado apresentou PID=PGID=SID próprio; isso dá uma fronteira
  clara para descendentes do servidor.
- Conflito em `8421` falha antes de criar outro processo.
- A UI não marca um launch como pronto por uma linha arbitrária de stdout; o
  contrato é o endpoint `/health`.
- O isolamento é uma defesa de ownership, não uma sandbox completa do
  `llama-server`; o binário continua executando com as permissões do usuário.
- Em Windows há um caminho específico de `CREATE_NEW_PROCESS_GROUP`; as
  evidências detalhadas desta ADR são do ambiente Linux validado.
