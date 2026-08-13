import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Fan,
  Loader2,
  MemoryStick,
  RefreshCcw,
  Thermometer,
  Zap,
} from "lucide-react";
import { api } from "../api/client";
import type { AmdGpu, AmdStatus, AmdTelemetryValue, SystemInfo } from "../api/types";

const HISTORY_LIMIT = 60;
const NA = "N/A";

type MemoryPoint = {
  at: number;
  vramUsed: number | null;
  vramTotal: number | null;
  ramUsed: number | null;
  ramTotal: number | null;
};

type PollResult = {
  status: AmdStatus;
  system: SystemInfo | null;
};

function finiteNumber(value: AmdTelemetryValue | undefined): number | null {
  if (value == null || (typeof value === "string" && value.trim() === "")) return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function isMissing(value: AmdTelemetryValue | undefined): boolean {
  return value == null || (typeof value === "string" && (value.trim() === "" || value.trim().toUpperCase() === NA));
}

function MissingValue() {
  return <span className="text-ink-500" aria-label="não disponível">{NA}</span>;
}

function displayValue(value: AmdTelemetryValue | undefined, unit = ""): React.ReactNode {
  if (isMissing(value)) return <MissingValue />;
  const numeric = finiteNumber(value);
  if (numeric != null) return <>{numeric.toLocaleString("pt-BR")}{unit}</>;
  return <>{String(value)}{unit && !String(value).toLowerCase().includes(unit.trim().toLowerCase()) ? unit : ""}</>;
}

function mib(value: AmdTelemetryValue | undefined): React.ReactNode {
  const parsed = finiteNumber(value);
  if (parsed == null) return <MissingValue />;
  return <>{parsed.toLocaleString("pt-BR")} MiB</>;
}

function hostTemperature(value: number | null): React.ReactNode {
  if (value == null || !Number.isFinite(value)) return <MissingValue />;
  return <>{value.toLocaleString("pt-BR", { maximumFractionDigits: 1 })} °C</>;
}

function percent(used: number | null, total: number | null): number | null {
  if (used == null || total == null || total <= 0) return null;
  return Math.max(0, Math.min(100, (used / total) * 100));
}

function makePoint(status: AmdStatus, system: SystemInfo | null): MemoryPoint {
  const available = status.available ? status : null;
  const ramTotal = system?.ram_total_mib ?? null;
  const ramAvailable = system?.ram_avail_mib ?? null;
  return {
    at: Date.now(),
    vramUsed: finiteNumber(available?.vram_used_mib),
    vramTotal: finiteNumber(available?.vram_total_mib),
    ramUsed: ramTotal != null && ramAvailable != null ? Math.max(ramTotal - ramAvailable, 0) : null,
    ramTotal,
  };
}

export function AmdPage() {
  const [status, setStatus] = useState<AmdStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [history, setHistory] = useState<MemoryPoint[]>([]);
  const [refreshToken, setRefreshToken] = useState(0);
  const generationRef = useRef(0);
  const inFlightRef = useRef<Promise<PollResult> | null>(null);

  useEffect(() => {
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    let cancelled = false;
    let timer: number | null = null;
    let cancelWait: (() => void) | null = null;

    const request = (): Promise<PollResult> => {
      if (inFlightRef.current) return inFlightRef.current;
      const pending = Promise.all([
        api.gpu(),
        api.system().catch(() => null),
      ]).then(([next, system]) => ({ status: next, system }));
      inFlightRef.current = pending;
      pending.then(
        () => { if (inFlightRef.current === pending) inFlightRef.current = null; },
        () => { if (inFlightRef.current === pending) inFlightRef.current = null; },
      );
      return pending;
    };

    const poll = async (showLoading: boolean): Promise<boolean> => {
      if (showLoading) setLoading(true);

      // Uma troca de aba, toggle ou refresh manual pode encontrar a leitura
      // anterior ainda em voo. Aguarda-a, descarta o resultado e só então
      // abre uma nova leitura para esta geração — nunca há requests concorrentes.
      while (inFlightRef.current) {
        const previous = inFlightRef.current;
        try {
          await previous;
        } catch {
          // A geração atual fará uma nova tentativa abaixo.
        }
        if (inFlightRef.current === previous) inFlightRef.current = null;
        if (cancelled || generationRef.current !== generation) return false;
      }

      try {
        const { status: next, system } = await request();
        if (cancelled || generationRef.current !== generation) return false;
        setStatus(next);
        setHistory((current) => [...current, makePoint(next, system)].slice(-HISTORY_LIMIT));
        setError(null);
        return true;
      } catch (e) {
        if (cancelled || generationRef.current !== generation) return false;
        setError((e as Error).message);
        return true;
      } finally {
        if (!cancelled && generationRef.current === generation) setLoading(false);
      }
    };

    const waitForNextPoll = () => new Promise<void>((resolve) => {
      cancelWait = () => {
        cancelWait = null;
        resolve();
      };
      timer = window.setTimeout(() => {
        timer = null;
        cancelWait = null;
        resolve();
      }, 2_000);
    });

    const loop = async (): Promise<void> => {
      let showLoading = true;
      while (!cancelled && generationRef.current === generation) {
        const completed = await poll(showLoading);
        if (!completed || cancelled || generationRef.current !== generation || !autoRefresh) return;
        await waitForNextPoll();
        showLoading = false;
      }
    };

    void loop();
    return () => {
      cancelled = true;
      generationRef.current += 1;
      if (timer != null) {
        window.clearTimeout(timer);
        timer = null;
        cancelWait?.();
      }
    };
  }, [autoRefresh, refreshToken]);

  return (
    <section className="bg-ink-900 border border-ink-800 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-ink-800 flex flex-wrap items-center gap-3">
        <h2 className="font-medium text-ink-100 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-accent-fg" /> AMD GPU
        </h2>
        <span className="text-xs text-ink-500">telemetria via sysfs</span>
        <div className="flex-1 min-w-4" />
        <label className="text-xs text-ink-400 flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(event) => setAutoRefresh(event.target.checked)}
            className="accent-emerald-500"
          />
          atualizar a cada 2 s
        </label>
        <button
          type="button"
          onClick={() => setRefreshToken((current) => current + 1)}
          className="text-xs px-2 py-1 rounded bg-ink-800 hover:bg-ink-700 flex items-center gap-1"
        >
          <RefreshCcw className="w-3 h-3" /> atualizar
        </button>
      </div>
      <div className="p-4">
        {loading && !status && (
          <div className="text-ink-400 text-sm flex items-center gap-2" role="status">
            <Loader2 className="w-3 h-3 animate-spin" /> consultando GPUs AMD…
          </div>
        )}
        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm mono" role="alert">
            Não foi possível consultar a telemetria AMD: {error}
          </div>
        )}
        {status && !status.available && (
          <div className="bg-amber-950/30 border border-amber-800 text-amber-200 rounded p-3 text-sm flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <div className="font-medium">Nenhuma GPU AMD disponível</div>
              <div className="text-xs text-amber-300/70 mt-1">{status.error}</div>
            </div>
          </div>
        )}
        {status?.available && <AmdTelemetry status={status} history={history} />}
      </div>
    </section>
  );
}

function AmdTelemetry({
  status,
  history,
}: {
  status: Extract<AmdStatus, { available: true }>;
  history: MemoryPoint[];
}) {
  const total = status.vram_total_mib;
  const used = status.vram_used_mib;
  const free = status.vram_free_mib;
  const usage = percent(used, total);
  const hostTemp = status.host_temp_c;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <Summary label="GPUs detectadas" value={String(status.gpu_count)} />
        <Summary label="VRAM total" value={mib(total)} />
        <Summary label="Temperatura do host" value={hostTemperature(hostTemp)} />
      </div>

      <div className="bg-ink-950 border border-ink-800 rounded p-4">
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs mb-2">
          <span className="text-ink-300 flex items-center gap-1"><MemoryStick className="w-3 h-3" /> VRAM agregada</span>
          <span className="mono tabular-nums text-ink-200">{mib(used)} / {mib(total)}</span>
        </div>
        <div
          role="progressbar"
          aria-label="Uso agregado de VRAM"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={usage ?? undefined}
          aria-valuetext={usage == null ? "N/A: uso de VRAM não disponível" : `${usage.toFixed(0)}% da VRAM em uso`}
          className="h-2 bg-ink-800 rounded overflow-hidden"
        >
          <div className={`h-full ${usage != null && usage >= 95 ? "bg-red-500" : usage != null && usage >= 80 ? "bg-amber-500" : "bg-emerald-500"}`} style={{ width: `${usage ?? 0}%` }} />
        </div>
        <div className="mt-2 flex justify-between text-[11px] text-ink-500 mono">
          <span>{free == null ? <MissingValue /> : <>{mib(free)} livres</>}</span>
          <span>{usage == null ? <MissingValue /> : `${usage.toFixed(0)}% usado`}</span>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <MemoryChart kind="vram" points={history} />
        <MemoryChart kind="ram" points={history} />
      </div>

      <div>
        <h3 className="font-medium text-ink-100">Dispositivos AMD</h3>
        <p className="text-xs text-ink-500 mt-1">Leituras por GPU; N/A indica que o sysfs não forneceu aquele dado.</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {status.gpus.map((gpu, index) => <GpuCard key={`${gpu.name}-${index}`} gpu={gpu} index={index} />)}
      </div>
    </div>
  );
}

function MemoryChart({ kind, points }: { kind: "vram" | "ram"; points: MemoryPoint[] }) {
  const isVram = kind === "vram";
  const values = points.map((point) => isVram ? point.vramUsed : point.ramUsed);
  const totals = points.map((point) => isVram ? point.vramTotal : point.ramTotal);
  const latest = [...values].reverse().find((value) => value != null) ?? null;
  const total = [...totals].reverse().find((value) => value != null) ?? null;
  const title = isVram ? "Histórico de VRAM" : "Histórico de RAM";
  const color = isVram ? "#34d399" : "#60a5fa";
  const hasData = values.some((value) => value != null);

  return (
    <section className="bg-ink-950 border border-ink-800 rounded p-4" aria-label={title}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h4 className="text-sm text-ink-200 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5" style={{ color }} /> {title}
          </h4>
          <p className="text-[11px] text-ink-500 mt-1">até {HISTORY_LIMIT} pontos · poll de 2 s</p>
        </div>
        <div className="text-right mono tabular-nums text-xs">
          <div className="text-ink-200">{latest == null ? <MissingValue /> : mib(latest)}</div>
          <div className="text-ink-600">{total == null ? <MissingValue /> : <>de {mib(total)}</>}</div>
        </div>
      </div>
      {!hasData ? (
        <div className="h-28 rounded border border-dashed border-ink-800 flex items-center justify-center text-xs text-ink-500">
          <span>N/A · histórico aguardando leitura</span>
        </div>
      ) : (
        <Sparkline values={values} total={total} color={color} label={title} />
      )}
    </section>
  );
}

function Sparkline({
  values,
  total,
  color,
  label,
}: {
  values: (number | null)[];
  total: number | null;
  color: string;
  label: string;
}) {
  const width = 440;
  const height = 112;
  const padding = 4;
  const max = Math.max(total ?? 0, ...values.filter((value): value is number => value != null), 1);
  const coordinates = values.map((value, index) => ({
    x: values.length <= 1 ? width / 2 : padding + (index / (values.length - 1)) * (width - padding * 2),
    y: value == null ? null : height - padding - (value / max) * (height - padding * 2),
  }));
  const segments: string[] = [];
  let segment = "";
  coordinates.forEach((point) => {
    if (point.y == null) {
      if (segment) segments.push(segment);
      segment = "";
    } else {
      segment += `${segment ? " L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
    }
  });
  if (segment) segments.push(segment);
  const latest = [...coordinates].reverse().find((point) => point.y != null);

  return (
    <svg
      className="w-full h-28 overflow-visible"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`${label}: ${values.filter((value) => value != null).length} leituras`}
    >
      <title>{label} em memória, janela limitada a {HISTORY_LIMIT} leituras</title>
      {[0.25, 0.5, 0.75].map((ratio) => (
        <line key={ratio} x1={padding} x2={width - padding} y1={height - ratio * height} y2={height - ratio * height} stroke="#272d36" strokeWidth="1" />
      ))}
      {segments.map((path, index) => <path key={index} d={path} fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />)}
      {latest?.y != null && <circle cx={latest.x} cy={latest.y} r="3.5" fill={color} stroke="#0d1117" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

function GpuCard({ gpu, index }: { gpu: AmdGpu; index: number }) {
  const total = finiteNumber(gpu["memory.total"]);
  const used = finiteNumber(gpu["memory.used"]);
  const usage = percent(used, total);
  return (
    <article className="bg-ink-950 border border-ink-800 rounded p-4 space-y-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h4 className="font-medium text-ink-100">{gpu.name || `GPU AMD ${index + 1}`}</h4>
          <div className="text-xs text-ink-500 mono">driver {displayValue(gpu.driver_version)}</div>
        </div>
        <span className="text-[10px] uppercase tracking-wider text-ink-500">GPU {index + 1}</span>
      </div>

      <div className="text-xs">
        <div className="flex justify-between mb-1 text-ink-300"><span>VRAM</span><span className="mono">{mib(gpu["memory.used"])} / {mib(gpu["memory.total"])}</span></div>
        <div
          role="progressbar"
          aria-label={`Uso de VRAM da ${gpu.name || `GPU AMD ${index + 1}`}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={usage ?? undefined}
          aria-valuetext={usage == null ? "N/A: uso de VRAM não disponível" : `${usage.toFixed(0)}% de VRAM em uso`}
          className="h-1.5 bg-ink-800 rounded overflow-hidden"
        >
          <div className="h-full bg-emerald-500" style={{ width: `${usage ?? 0}%` }} />
        </div>
        <div className="mt-1 text-ink-500">{mib(gpu["memory.free"])} livres</div>
      </div>

      <MetricGroup title="Temperatura" icon={<Thermometer className="w-3.5 h-3.5 text-orange-300" />} items={[
        ["Edge", gpu["temperature.gpu"], " °C"],
        ["Memória", gpu["temperature.memory"], " °C"],
        ["Hotspot", gpu["temperature.hotspot"], " °C"],
        ["Limite GPU", gpu["temperature.gpu.limit"], " °C"],
        ["Folga térmica", gpu["temperature.gpu.tlimit"], " °C"],
      ]} />
      <MetricGroup title="Utilização e ventilação" icon={<Fan className="w-3.5 h-3.5 text-sky-300" />} items={[
        ["GPU", gpu["utilization.gpu"], "%"],
        ["Memória", gpu["utilization.memory"], "%"],
        ["Fan", gpu["fan.speed"], " RPM"],
      ]} />
      <MetricGroup title="Energia e clocks" icon={<Zap className="w-3.5 h-3.5 text-amber-300" />} items={[
        ["Power draw", gpu["power.draw"], " W"],
        ["Limite", gpu["power.limit"], " W"],
        ["Clock GPU", gpu["clocks.sm"], " MHz"],
        ["Clock memória", gpu["clocks.mem"], " MHz"],
      ]} />
    </article>
  );
}

function MetricGroup({
  title,
  icon,
  items,
}: {
  title: string;
  icon: React.ReactNode;
  items: [string, AmdTelemetryValue | undefined, string][];
}) {
  return (
    <div className="border-t border-ink-800 pt-3">
      <div className="text-[11px] uppercase tracking-wider text-ink-500 flex items-center gap-1.5 mb-2">{icon}{title}</div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
        {items.map(([label, value, unit]) => (
          <div key={label} className="flex items-center justify-between gap-2 min-w-0">
            <dt className="text-ink-500 truncate">{label}</dt>
            <dd className="mono tabular-nums text-ink-200 text-right">{displayValue(value, unit)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Summary({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-ink-950 border border-ink-800 rounded p-3">
      <div className="text-[11px] uppercase tracking-wider text-ink-500">{label}</div>
      <div className="mt-1 text-lg mono tabular-nums text-ink-100">{value}</div>
    </div>
  );
}
