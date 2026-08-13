# ADR 0009 — Paridade de telemetria AMD via sysfs

## Status

Aceita e implementada; a implementação foi concluída na Fase 2 e esta ADR é a
consolidação documental da Fase 7.

## Contexto

O contrato anterior expunha essencialmente VRAM: total, uso e livre agregados,
com `mem_info_vram_used` opcional. Temperatura e utilização eram a limitação
histórica registrada na [Nota de supersessão da ADR 0002:L7-L12](0002-sysfs-em-vez-de-rocm-smi.md#L7-L12).
A implementação da telemetria na API e na UI foi requerida, concluída e
validada na Fase 2, conforme a evidência de API e UI em
evidência registrada no repositório privado (logs/final/2.log:L1-L2) e
evidência registrada no repositório privado (logs/final/2.log:L41-L91). A Fase 7
documenta e consolida essa decisão, sem ser a fase de implementação.

O objetivo é paridade de observabilidade, não equivalência semântica com um
monitor ROCm específico. O sysfs pode não expor um sensor em determinado host;
nesse caso a ausência precisa ser representada, e não estimada ou escondida.

## Decisão

### Descoberta e validação AMD

Em cada leitura, `app/api/core/amd.py` enumera novamente
`/sys/class/drm/card*/device` em ordem lexicográfica. Um card só é válido quando
os dois campos obrigatórios são legíveis e válidos: `vendor` é exatamente
`0x1002` e `mem_info_vram_total` é um inteiro positivo. Somente vendor AMD
ausente/ilegível/incorreto ou total ausente/ilegível/inválido/não positivo
exclui o card; um sysfs parcial não exclui automaticamente a GPU. A lógica dos
campos obrigatórios está em
[`app/api/core/amd.py:L193-L214`](../../app/api/core/amd.py#L193-L214) e
[`app/api/core/amd.py:L38-L43`](../../app/api/core/amd.py#L38-L43).

`mem_info_vram_used` é opcional: sua ausência mantém o card válido, produz
`N/A` para uso/livre do card e `None` para uso/livre agregados. Campos de
sensor ou hwmon ausentes/ilegíveis também mantêm o card válido e produzem
`N/A` por GPU, enquanto o sensor térmico do host produz `null`. Essas semânticas
estão em [`app/api/core/amd.py:L253-L295`](../../app/api/core/amd.py#L253-L295) e
[`app/api/core/amd.py:L171-L190`](../../app/api/core/amd.py#L171-L190), com a
leitura opcional de sensores em
[`app/api/core/amd.py:L64-L117`](../../app/api/core/amd.py#L64-L117). Os
invariantes de card e uso ausente são cobertos por
[`app/api/tests/test_amd.py:L59-L76`](../../app/api/tests/test_amd.py#L59-L76).

### Agregação e conversão de memória

Os contadores `mem_info_vram_total` e `mem_info_vram_used` são lidos em bytes.
O módulo define um MiB como `1024 * 1024`; soma os bytes por campo e só então
converte por divisão inteira. Para livre por card, calcula
`max(total_bytes - used_bytes, 0)` antes da conversão. Se algum uso faltar,
total continua disponível e uso/livre agregados passam a `None`; o card
individual mostra `N/A` para uso/livre. Isso é implementado em
[`app/api/core/amd.py:L22-L24`](../../app/api/core/amd.py#L22-L24) e
[`app/api/core/amd.py:L217-L225`](../../app/api/core/amd.py#L217-L225), e a
ordem de subtração é verificada em
[`app/api/tests/test_amd.py:L39-L56`](../../app/api/tests/test_amd.py#L39-L56).

### Sensores opcionais e mapeamentos

Cada card preserva `N/A` quando a fonte não existe, não pode ser lida ou contém
um número inválido. A leitura de temperatura procura labels `edge`, `mem` e
`junction` em `hwmon`; `edge_max` é preferido a `edge_crit`, e a folga térmica
é `limite - temperatura atual`. O sensor térmico do host usa `psutil`, prioriza
`coretemp`, `k10temp`, `acpitz` e `cpu_thermal`, e retorna `null` quando não há
leitura. As fontes e as conversões do backend estão em
[`app/api/core/amd.py:L64-L117`](../../app/api/core/amd.py#L64-L117),
[`app/api/core/amd.py:L150-L190`](../../app/api/core/amd.py#L150-L190) e
[`app/api/core/amd.py:L253-L295`](../../app/api/core/amd.py#L253-L295).

| Campo do card | Fonte sysfs/hwmon e unidade do contrato |
|---|---|
| `name`, `vendor` | Nome derivado do card e vendor AMD validado. |
| `memory.total`, `memory.used`, `memory.free` | `mem_info_vram_total`/`mem_info_vram_used`, bytes convertidos para MiB conforme a agregação acima. |
| `temperature.gpu`, `temperature.memory`, `temperature.hotspot` | Labels `edge`, `mem`, `junction`; milicelsius convertidos para °C. |
| `temperature.gpu.limit`, `temperature.gpu.tlimit` | `edge_max` ou `edge_crit`; `tlimit` é limite menos temperatura atual, em °C. |
| `fan.speed` | `fan1_input`, em RPM. |
| `utilization.gpu`, `utilization.memory` | `gpu_busy_percent` e `mem_busy_percent`, em porcentagem. |
| `power.draw`, `power.limit` | `power1_average` e `power1_cap`, microwatts convertidos para W. |
| `clocks.sm`, `clocks.mem` | Entrada marcada com `*` em `pp_dpm_sclk` e `pp_dpm_mclk`, em MHz, serializada como `Mhz` pelo backend. |
| `driver_version` | `/sys/module/amdgpu/version`, com fallback para `platform.release()`. |

O formato público preserva os valores por card como texto (incluindo `N/A`) e
o sensor térmico do host como número ou `null`; esses tipos estão espelhados em
[`app/src/api/types.ts:L313-L355`](../../app/src/api/types.ts#L313-L355). A UI
renderiza ausência como `N/A`, sem converter ausência em zero, conforme
[`app/src/components/AmdPage.tsx:L32-L61`](../../app/src/components/AmdPage.tsx#L32-L61).

### API, polling e histórico

`GET /api/gpu` devolve diretamente `amd.amd_status()`, que inclui o envelope
`available`, `error` quando indisponível, `gpu_count`, `gpus`,
`vram_total_mib`, `vram_used_mib`, `vram_free_mib` e `host_temp_c` quando há
cards válidos. A rota e a construção da resposta estão em
[`app/api/server.py:L298-L312`](../../app/api/server.py#L298-L312) e
[`app/api/core/amd.py:L298-L320`](../../app/api/core/amd.py#L298-L320).

Na página AMD, o frontend consulta `/api/gpu` e `/api/system` imediatamente e,
quando o refresh automático está ativo, espera 2 segundos entre leituras. O
histórico é estado local React, mantém no máximo 60 pontos e registra somente
VRAM/RAM; não existe uma série temporal persistida no backend. As regras estão
em [`app/src/components/AmdPage.tsx:L16-L24`](../../app/src/components/AmdPage.tsx#L16-L24),
[`app/src/components/AmdPage.tsx:L92-L178`](../../app/src/components/AmdPage.tsx#L92-L178)
e [`app/src/components/AmdPage.tsx:L290-L320`](../../app/src/components/AmdPage.tsx#L290-L320).

### Verificação de coerência

O script de verificação faz leituras consecutivas da API e do sysfs. As
tolerâncias abaixo são uma política dinâmica para a diferença entre essas duas
leituras, não defaults de hardware nem limites físicos:

| Leitura dinâmica | Tolerância |
|---|---:|
| VRAM usada/livre | 2 MiB |
| Temperaturas GPU, memória, hotspot, folga térmica e host | 2 °C |
| Fan | 100 RPM |
| Utilização GPU e memória | 5 pontos percentuais |
| Power draw | 2 W |
| Clocks GPU e memória | 100 MHz |

Esses valores devem permanecer exatamente iguais a
`DYNAMIC_TOLERANCES` em
[`scripts/verify-amd-telemetry.py:L39-L53`](../../scripts/verify-amd-telemetry.py#L39-L53).
O script obtém a resposta da API e, na mesma verificação, descobre os cards e
faz a leitura sysfs; portanto a política é dinâmica entre leituras consecutivas,
não default de hardware, conforme
[`scripts/verify-amd-telemetry.py:L282-L300`](../../scripts/verify-amd-telemetry.py#L282-L300).
VRAM total, limites dos sensores, labels, unidades e driver são verificações
exatas. `temperature.gpu.tlimit` não é exato: recebe a tolerância dinâmica de
2 °C na comparação do campo em
[`scripts/verify-amd-telemetry.py:L313-L315`](../../scripts/verify-amd-telemetry.py#L313-L315),
e a coerência `tlimit = limite - atual` também aceita a entrada de 2 °C em
[`scripts/verify-amd-telemetry.py:L316-L320`](../../scripts/verify-amd-telemetry.py#L316-L320).

Como evidência, uma única captura registrou `gpu_count=1`, VRAM
`24560/26/24533 MiB`, temperaturas GPU/memória/hotspot `42/54/47 °C`, limite e
folga `100/58 °C`, fan `0 RPM`, utilização `0/0%`, power `17/272 W`, clocks
`0/96 Mhz`, driver `6.8.0-51-generic` e host `68.125 °C` em
evidência registrada no repositório privado (logs/final/2.log:L41-L42) e
evidência registrada no repositório privado (logs/final/2.log:L51-L74). São valores daquela
run capturada, não defaults, tolerâncias ou expectativas universais de hardware.

## Consequências

- `rocm-smi` não é necessário para iniciar o backend ou responder à API; a
  decisão de fonte continua em sysfs, agora com DRM, hwmon e `psutil` opcionais.
- A API pode entregar uma visão detalhada por GPU e agregados de memória sem
  afirmar que todos os sensores existem em todos os hosts.
- O histórico da UI é útil para observação durante a sessão, mas reiniciar a
  página ou o backend perde os pontos; análises históricas exigem um coletor
  externo.
- A enumeração e a soma suportam múltiplos cards em nível de dados, mas não
  provam distribuição de pesos, KV ou compute pelo `llama.cpp`.
- A evidência de hardware desta fase é limitada a uma captura com uma única
  GPU AMD (`gpu_count=1`) em evidência registrada no repositório privado (logs/final/2.log:L41-L46).
  O cenário de múltiplos cards é teste unitário/contratual, não validação
  física multi-GPU, conforme [`app/api/tests/test_amd.py:L19-L36`](../../app/api/tests/test_amd.py#L19-L36).
