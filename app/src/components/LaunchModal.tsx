import React from "react";
import { Square, XCircle, CheckCircle, AlertTriangle, ArrowDownUp, RotateCw, Play } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, eventsStream } from "../api/client";
import type { ActiveLaunchConfig, LaunchEvent, LaunchEventCursor } from "../api/types";
import { isRouterConfig } from "../api/types";
import { Modal } from "./Modal";

type Props = {
  open: boolean;
  launchId: string | null;
  config: ActiveLaunchConfig | null;
  onClose: () => void;
  onCancel: () => Promise<void>;
  onTerminal: (launchId: string) => void;
};

// Modal de execução: assina o SSE `/api/launch/{id}/events`, renderiza stdout
// scrollado + faixa de eventos de degrade/restart/giveup destacados. Quando
// o runner termina com hint=lms_fallback_available (mmproj quebrado),
// oferece botão pra tentar via LM Studio.
export function LaunchModal({ open, launchId, config, onClose, onCancel, onTerminal }: Props) {
  const [lines, setLines] = useState<{ kind: string; text: string; icon?: React.ReactNode }[]>([]);
  const [done, setDone] = useState(false);
  const [currentCfg, setCurrentCfg] = useState<ActiveLaunchConfig | null>(config);
  const [lastEventId, setLastEventId] = useState<LaunchEventCursor>(null);
  const [lmsHint, setLmsHint] = useState(false);
  const [lmsLoading, setLmsLoading] = useState(false);
  const [lmsResult, setLmsResult] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<LaunchEventCursor>(null);
  const seenSeqsRef = useRef<Set<number>>(new Set());
  const seenHistoryGapsRef = useRef<Set<string>>(new Set());
  const terminalNotifiedRef = useRef(false);

  // Config refreshes must not reset the transcript. Only a new launch id does.
  useEffect(() => {
    setLines([]);
    setDone(false);
    setCurrentCfg(config);
    cursorRef.current = null;
    seenSeqsRef.current = new Set();
    seenHistoryGapsRef.current = new Set();
    terminalNotifiedRef.current = false;
    setLastEventId(null);
    setLmsHint(false);
    setLmsResult(null);
  }, [launchId]);

  useEffect(() => {
    if (config) setCurrentCfg(config);
  }, [config]);

  useEffect(() => {
    if (!open || !launchId) return;
    let es: EventSource | null = null;
    let cancelled = false;
    let reconnectTimer: number | null = null;
    let connect: () => Promise<void>;
    let removeHistoryGapListener: (() => void) | null = null;

    const closeStream = () => {
      removeHistoryGapListener?.();
      removeHistoryGapListener = null;
      es?.close();
      es = null;
    };

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimer !== null) return;
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        void connect();
      }, 500);
    };

    connect = async () => {
      if (cancelled) return;
      try {
        const stream = await eventsStream(launchId, cursorRef.current);
        if (cancelled) {
          stream.close();
          return;
        }

        const handleMessage = (
          data: string,
          lastEventId: string,
          namedType?: "history_gap",
        ) => {
          if (cancelled) return;
          try {
            const parsed = JSON.parse(data) as LaunchEvent | Record<string, unknown>;
            // A named SSE event may carry only its payload. Normalize that
            // form before sending it through the same lifecycle pipeline.
            const event = namedType === "history_gap" &&
              parsed !== null && typeof parsed === "object" &&
              !("type" in parsed)
              ? { ...parsed, type: "history_gap" } as LaunchEvent
              : parsed as LaunchEvent;
            if (event.type === "history_gap") {
              // A replay gap describes the cursor requested by the client,
              // so its id can equal the current cursor. Handle it before the
              // ordinary sequence guard and keep it out of that sequence
              // state; only the gap-specific key deduplicates it.
              const gapKey = lastEventId.trim()
                ? `seq:${lastEventId}`
                : `payload:${data}`;
              if (seenHistoryGapsRef.current.has(gapKey)) return;
              seenHistoryGapsRef.current.add(gapKey);
              handleEvent(event);
              return;
            }

            const seq = parseEventId(lastEventId);
            if (seq !== null) {
              if (
                seenSeqsRef.current.has(seq) ||
                (cursorRef.current !== null && seq <= cursorRef.current)
              ) return;
              seenSeqsRef.current.add(seq);
              cursorRef.current = seq;
              setLastEventId(seq);
            }
            handleEvent(event);
          } catch {
            /* ignore malformed data */
          }
        };

        es = stream;
        es.onmessage = (e) => handleMessage(e.data, e.lastEventId);
        const handleHistoryGap = (e: MessageEvent<string>) =>
          handleMessage(e.data, e.lastEventId, "history_gap");
        es.addEventListener("history_gap", handleHistoryGap);
        removeHistoryGapListener = () => es?.removeEventListener("history_gap", handleHistoryGap);
        es.onerror = () => {
          if (cancelled || terminalNotifiedRef.current) return;
          closeStream();
          // A dropped SSE connection is not a terminal launch event.
          scheduleReconnect();
        };
      } catch {
        scheduleReconnect();
      }
    };

    void connect();
    return () => {
      cancelled = true;
      closeStream();
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
    };
  }, [launchId, open]);

  function parseEventId(raw: string): number | null {
    if (!raw.trim()) return null;
    const value = Number(raw);
    return Number.isInteger(value) && value >= 0 ? value : null;
  }

  function notifyTerminal() {
    if (!launchId || terminalNotifiedRef.current) return;
    terminalNotifiedRef.current = true;
    onTerminal(launchId);
  }

  function handleEvent(ev: LaunchEvent) {
    setLines((prev) => {
      const next = [...prev];
      switch (ev.type) {
        case "start":
          next.push({ kind: "info", text: `tentativa #${ev.attempt}`, icon: <Play className="w-3 h-3" /> });
          next.push({ kind: "cmd",  text: ev.cmd });
          setCurrentCfg(ev.config);
          break;
        case "stdout":
          next.push({ kind: "stdout", text: ev.line });
          break;
        case "load_ok":
          next.push({ kind: "ok", text: `servidor carregou (tentativa #${ev.attempt}) — config salva`, icon: <CheckCircle className="w-3 h-3" /> });
          break;
        case "exit":
          next.push({ kind: "info", text: `process exit rc=${ev.rc} (tentativa #${ev.attempt})` });
          break;
        case "failure":
          next.push({ kind: "fail", text: `${ev.category}: ${ev.excerpt}`, icon: <XCircle className="w-3 h-3" /> });
          break;
        case "degrade":
          next.push({ kind: "degrade", text: `degrade: ${ev.description}`, icon: <ArrowDownUp className="w-3 h-3" /> });
          setCurrentCfg(ev.config);
          break;
        case "restart":
          next.push({ kind: "info", text: `reiniciando (#${ev.attempt}) em ${ev.backoff}s…`, icon: <RotateCw className="w-3 h-3" /> });
          break;
        case "manual_restart":
          next.push({ kind: "degrade", text: `restart manual — subindo de novo (#${ev.attempt}), mesma config`, icon: <RotateCw className="w-3 h-3" /> });
          break;
        case "giveup":
          next.push({ kind: "fail", text: `desistiu: ${ev.reason}${ev.failure ? ` (${ev.failure})` : ""}`, icon: <Square className="w-3 h-3" /> });
          setDone(true);
          notifyTerminal();
          if (ev.hint === "lms_fallback_available") setLmsHint(true);
          break;
        case "done":
          next.push({ kind: "ok", text: `encerrou normalmente (tentativa #${ev.attempt})`, icon: <CheckCircle className="w-3 h-3" /> });
          setDone(true);
          notifyTerminal();
          break;
        case "history_gap":
          next.push({
            kind: "gap",
            text: "histórico parcial — alguns eventos anteriores já não estão disponíveis",
            icon: <AlertTriangle className="w-3 h-3" />,
          });
          break;
      }
      return next;
    });
    // autoscroll
    setTimeout(() => {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    }, 0);
  }

  const handleCancel = async () => {
    if (!launchId) return;
    setDone(true);
    setLines((prev) => [...prev, { kind: "fail", text: "cancelado pelo usuário", icon: <Square className="w-3 h-3" /> }]);
    await onCancel();
  };

  const handleRestart = async () => {
    if (!launchId) return;
    setLines((prev) => [...prev, { kind: "info", text: "restart manual solicitado…", icon: <RotateCw className="w-3 h-3" /> }]);
    try {
      await api.restartLaunch(launchId);
    } catch (e) {
      setLines((prev) => [...prev, { kind: "fail", text: `falha ao reiniciar: ${(e as Error).message}`, icon: <XCircle className="w-3 h-3" /> }]);
    }
  };

  const handleLmsFallback = async () => {
    if (!currentCfg || isRouterConfig(currentCfg)) return;
    setLmsLoading(true);
    setLines((prev) => [...prev, { kind: "info", text: `tentando carregar via LM Studio…`, icon: <Play className="w-3 h-3" /> }]);
    try {
      const r = await api.lmsLoad({
        model: currentCfg.model,
        context_window: currentCfg.context_window,
        parallel_slots: currentCfg.parallel_slots,
      });
      if (r.ok) {
        setLines((prev) => [
          ...prev,
          { kind: "ok", text: `LM Studio carregou: ${r.alias} (key: ${r.key})`, icon: <CheckCircle className="w-3 h-3" /> },
        ]);
        setLmsResult("ok");
      } else {
        setLines((prev) => [
          ...prev,
          { kind: "fail", text: `LM Studio falhou: ${r.error ?? r.load_output ?? "erro desconhecido"}`, icon: <XCircle className="w-3 h-3" /> },
        ]);
        setLmsResult("fail");
      }
    } catch (e) {
      setLines((prev) => [
        ...prev,
        { kind: "fail", text: `erro chamando lms: ${(e as Error).message}`, icon: <XCircle className="w-3 h-3" /> },
      ]);
      setLmsResult("fail");
    } finally {
      setLmsLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      closable={done}
      width="max-w-5xl"
      testId="launch-modal"
      title={`Launch ${launchId ? `#${launchId.slice(0, 6)}` : ""}`}
      footer={
        <>
          {!done && (
            <>
              <button
                onClick={handleRestart}
                className="px-3 py-1.5 text-sm rounded bg-amber-900/60 hover:bg-amber-800 text-amber-100 flex items-center gap-1"
                title="Mata e sobe o llama-server de novo com a mesma config (mesma porta). O cliente reconecta."
              >
                <RotateCw className="w-3.5 h-3.5" />Reiniciar
              </button>
              <button
                onClick={handleCancel}
                data-testid="launch-modal-cancel"
                className="px-3 py-1.5 text-sm rounded bg-red-900/60 hover:bg-red-800 text-red-100 flex items-center gap-1"
              >
                <XCircle className="w-3.5 h-3.5" />Cancelar
              </button>
            </>
          )}
          {lmsHint && !lmsResult && (
            <button
              onClick={handleLmsFallback}
              disabled={lmsLoading}
              className="px-3 py-1.5 text-sm rounded bg-amber-900/60 hover:bg-amber-800 text-amber-100 flex items-center gap-1 disabled:opacity-50"
              title="mmproj falhou no llama-server — tentar carregar via LM Studio"
            >
              <Play className="w-3 h-3" /> tentar via LM Studio
            </button>
          )}
          <button
            onClick={onClose}
            data-testid="launch-modal-dismiss"
            className="px-3 py-1.5 text-sm rounded bg-ink-800 hover:bg-ink-700 flex items-center gap-1"
          >
            {done ? "Fechar" : "Esconder"}
          </button>
        </>
      }
    >
      <div className="p-4 space-y-3">
        {currentCfg && (isRouterConfig(currentCfg) ? (
          <div className="text-xs text-ink-400 mono">
            <span className="text-emerald-400">router</span> ·
            <span className="text-accent-fg"> {currentCfg.backend}</span> ·
            <span className="text-ink-200"> {currentCfg.model_ids.length}</span> modelos:{" "}
            <span className="text-ink-200">{currentCfg.model_ids.join(", ")}</span>
            <span className="text-ink-500"> — o client escolhe pelo campo "model" do request</span>
          </div>
        ) : (
          <div className="text-xs text-ink-400 mono">
            <span className="text-accent-fg">{currentCfg.backend}</span> ·
            ctx <span className="text-ink-200">{currentCfg.context_window.toLocaleString()}</span> ·
            kv <span className="text-ink-200">{currentCfg.kv_cache}</span> ·
            ngl <span className="text-ink-200">{currentCfg.gpu_layers}</span> ·
            np <span className="text-ink-200">{currentCfg.parallel_slots}</span>
            {currentCfg.n_cpu_moe > 0 && <> · ncmoe <span className="text-ink-200">{currentCfg.n_cpu_moe}</span></>}
          </div>
        ))}
        {lines.some((l) => l.text === "(reanexado)") && (
          <div className="bg-amber-950/30 border border-amber-800/60 text-amber-200 text-xs rounded p-2 flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>
              Este servidor foi <b>reanexado</b> de uma sessão anterior do app —
              os logs ao vivo só começam num próximo launch. O botão de Cancelar
              continua funcionando.
            </span>
          </div>
        )}

        <div
          ref={scrollRef}
          className="bg-ink-950 border border-ink-800 rounded mono text-xs h-[60vh] overflow-y-auto p-3 leading-relaxed"
        >
          {lastEventId !== null && (
            <div className="text-[10px] text-ink-600 mb-1">cursor #{lastEventId}</div>
          )}
          {lines.length === 0 && (
            <div className="text-ink-500 italic flex items-center gap-1.5">
              <Play className="w-3 h-3" />aguardando eventos…
            </div>
          )}
          {lines.map((l, i) => (
            <div
              key={i}
              className={`flex items-start gap-1.5 ${
                l.kind === "stdout"  ? "text-ink-300" :
                l.kind === "ok"      ? "text-emerald-400" :
                l.kind === "fail"    ? "text-red-400" :
                l.kind === "degrade" || l.kind === "gap" ? "text-amber-400" :
                l.kind === "cmd"     ? "text-purple-300 break-all" :
                                       "text-ink-400"
              }`}
            >
              {l.icon && <span className="mt-0.5 shrink-0">{l.icon}</span>}
              <span>{l.text}</span>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}
