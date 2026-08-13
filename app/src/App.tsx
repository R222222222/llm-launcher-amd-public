import { useCallback, useEffect, useRef, useState } from "react";
import { Settings, HardDrive, Download, Cpu, Loader2, SlidersHorizontal, Plug } from "lucide-react";
import { api, downloadEventsStream } from "./api/client";
import type {
  ActiveLaunchConfig, BackendStatus, Estimate, HfDownloadEvent, HfDownloadPlan,
  HfDownloadTerminal, HfExpectedFile, HfRevision,
  LaunchConfig, ModelInfo, ModelUpdate, Options, SystemInfo,
} from "./api/types";

export type DownloadProgress = { dl: number; total: number; speed: number; done?: boolean };
import { ConfigEditor } from "./components/ConfigEditor";
import { ConfigGrid } from "./components/ConfigGrid";
import { DownloadPage } from "./components/DownloadPage";
import { Header } from "./components/Header";
import { LaunchModal } from "./components/LaunchModal";
import { MCPPage } from "./components/MCPPage";
import { ModelsPage } from "./components/ModelsPage";
import { AmdPage } from "./components/AmdPage";
import { SettingsPage } from "./components/SettingsPage";
import { Tabs } from "./components/Tabs";

type TabId = "configs" | "models" | "download" | "mcp" | "amd" | "settings";

type LocalLaunch = {
  id: string;
  config: ActiveLaunchConfig;
  active: boolean;
  origin: "owned" | "attached";
};

const FULL_SHA256 = /^[0-9a-f]{64}$/i;

export default function App() {
  const [system, setSystem]       = useState<SystemInfo | null>(null);
  const [backends, setBackends]   = useState<BackendStatus[]>([]);
  const [options, setOptions]     = useState<Options | null>(null);
  const [models, setModels]       = useState<ModelInfo[]>([]);
  // Atualizações de GGUF por path do modelo. `null` na entrada = ainda checando.
  const [updates, setUpdates]     = useState<Record<string, ModelUpdate>>({});
  const [updatesChecking, setUpdatesChecking] = useState(false);
  const [configs, setConfigs]     = useState<LaunchConfig[]>([]);
  const [estimates, setEstimates] = useState<Record<string, Estimate>>({});
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);

  const [tab, setTab] = useState<TabId>("configs");

  const [editorOpen, setEditorOpen] = useState(false);
  const [editorInitial, setEditorInitial] = useState<Partial<LaunchConfig> | null>(null);

  // Keep identity, config, activity, and origin together so refresh cannot
  // erase the history of a launch created by this page.
  const [localLaunch, setLocalLaunch] = useState<LocalLaunch | null>(null);
  const [launchModalOpen, setLaunchModalOpen] = useState(false);

  // ─── Download state: vive aqui (não na DownloadPage) pra sobreviver à
  // troca de aba — sem isso, navegar pra outra aba unmounta a página e
  // mata o EventSource, perdendo o progresso (download segue no backend).
  const [downloadId,     setDownloadId]     = useState<string | null>(null);
  const [downloadPlan,   setDownloadPlan]   = useState<HfDownloadPlan | null>(null);
  const [downloadEvents, setDownloadEvents] = useState<HfDownloadEvent[]>([]);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgress>>({});
  const [downloadTerminal, setDownloadTerminal] = useState<HfDownloadTerminal | null>(null);
  const [downloadCancelling, setDownloadCancelling] = useState(false);
  const [downloadError,  setDownloadError]  = useState<string | null>(null);
  const downloadGeneration = useRef(0);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const [sys, bks, opts, mdls, cfgs, active] = await Promise.all([
        api.system(),
        api.backends(),
        api.options(),
        api.models(),
        api.listConfigs(),
        api.listLaunches(),
      ]);
      setSystem(sys);
      setBackends(bks);
      setOptions(opts);
      setModels(mdls);
      setConfigs(cfgs);

      // Reconcile with the server without destroying a local launch record.
      // A launch created here stays available as history when it leaves the
      // active list; an attached launch may be discarded and re-attached later.
      setLocalLaunch((current) => {
        if (current) {
          const mine = active.find((a) => a.launch_id === current.id);
          if (mine) return { ...current, active: true, config: mine.config };
          return current.origin === "owned"
            ? { ...current, active: false }
            : null;
        }
        const first = active[0];
        return first
          ? { id: first.launch_id, config: first.config, active: true, origin: "attached" }
          : null;
      });
      // Estimativa em batch pra alimentar a coluna de status do grid.
      // Roda em segundo plano — não bloqueia o render inicial.
      if (cfgs.length > 0) {
        api.estimateMany(cfgs.map((c) => ({
          model: c.model, backend: c.backend,
          context_window: c.context_window, kv_cache: c.kv_cache,
          parallel_slots: c.parallel_slots, gpu_layers: c.gpu_layers,
          n_cpu_moe: c.n_cpu_moe, mmproj: c.mmproj,
          cache_ram: c.cache_ram, mode: "server",
        }))).then((results) => {
          const map: Record<string, Estimate> = {};
          // Resposta vem na MESMA ordem dos items enviados — chaveia pelo id
          // da config (mesma chave que o grid usa via cfgKey), não por
          // model|backend, que colide com configs duplicadas e não bate com
          // o lookup do grid quando a config tem id.
          results.forEach((r, i) => {
            if (r.ok && r.estimate) {
              const c = cfgs[i];
              map[c.id ?? `${c.model}|${c.backend}`] = r.estimate;
            }
          });
          setEstimates(map);
        }).catch(() => { /* silencioso — só apaga as bolinhas */ });
      } else {
        setEstimates({});
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // The backend can be upgraded or downgraded independently of the frontend.
  // If MCP disappears while its tab is selected, return to the stable default
  // without mounting MCPPage (which is what performs MCP requests).
  const mcpEnabled = options?.features?.mcp === true;
  useEffect(() => {
    if (!mcpEnabled && tab === "mcp") setTab("configs");
  }, [mcpEnabled, tab]);

  // Checagem de atualização dos GGUFs. Roda na inicialização e sempre que o
  // CONJUNTO de modelos muda (baixou/apagou) — não a cada refresh (launch/save
  // recarregam a lista mas não mexem em quais modelos existem). É best-effort e
  // em segundo plano: rede offline só deixa os status como 'unknown'.
  const modelsKey = models.map((m) => m.path).join("|");
  useEffect(() => {
    if (models.length === 0) { setUpdates({}); return; }
    let cancelled = false;
    setUpdatesChecking(true);
    api.modelsUpdates(false)
      .then(({ results }) => {
        if (cancelled) return;
        const map: Record<string, ModelUpdate> = {};
        for (const r of results) map[r.path] = r;
        setUpdates(map);
      })
      .catch(() => { /* silencioso — sem badges, app segue normal */ })
      .finally(() => { if (!cancelled) setUpdatesChecking(false); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelsKey]);

  // Re-checagem profunda de UM modelo (hasheia pra confirmar). Botão manual.
  const handleRecheckModel = useCallback(async (path: string): Promise<ModelUpdate> => {
    const r = await api.checkModelUpdate(path, true);
    setUpdates((prev) => ({ ...prev, [r.path]: r }));
    return r;
  }, []);

  // Poll periódico do system info pra barras se atualizarem enquanto edita.
  // 2 s casa com a cadência da telemetria — sem polling extra
  // desperdiçado e sem ficar parado.
  useEffect(() => {
    const id = setInterval(() => api.system().then(setSystem).catch(() => {}), 2_000);
    return () => clearInterval(id);
  }, []);

  // SSE de download mora aqui (App) pra não morrer ao trocar de aba.
  useEffect(() => {
    if (!downloadId) return;
    let cancelled = false;
    let es: EventSource | null = null;
    let streamTerminal = false;

    const closeStream = () => {
      es?.close();
      es = null;
    };

    const finish = (terminal: HfDownloadTerminal, message?: string) => {
      if (cancelled || streamTerminal) return;
      streamTerminal = true;
      setDownloadTerminal(terminal);
      setDownloadCancelling(false);
      if (message) setDownloadError(message);
      closeStream();
      if (terminal === "done") refresh();
    };

    (async () => {
      try {
        const stream = await downloadEventsStream(downloadId);
        if (cancelled) {
          stream.close();
          return;
        }
        es = stream;
        es.onmessage = (msg) => {
          if (cancelled || streamTerminal) return;
          try {
            const ev = JSON.parse(msg.data) as HfDownloadEvent;
            // Progress NÃO entra no array de eventos: chegam ~2.5/s durante o
            // download inteiro, o array crescia sem limite e cada tick
            // re-renderizava o App todo — a UI "travava" enquanto baixava.
            // A barra de progresso usa só o map downloadProgress.
            if (ev.type !== "progress") {
              setDownloadEvents((prev) => [...prev, ev]);
            }
            if (ev.type === "progress") {
              setDownloadProgress((p) => ({
                ...p,
                [ev.rel]: { dl: ev.downloaded, total: ev.total, speed: ev.speed },
              }));
            } else if (ev.type === "file_done") {
              setDownloadProgress((p) => ({
                ...p,
                [ev.rel]: { ...(p[ev.rel] ?? { dl: 0, total: 0, speed: 0 }), done: true },
              }));
            } else if (ev.type === "done") {
              finish("done");
            } else if (ev.type === "cancelled") {
              finish("cancelled");
            } else if (ev.type === "error") {
              finish("error", ev.message);
            }
          } catch { /* ignore malformed events */ }
        };
        es.onerror = () => {
          if (cancelled || streamTerminal) return;
          // A transport failure is an error, never a successful completion.
          finish("error", "o stream do download caiu antes de um evento terminal");
        };
      } catch (e) {
        finish("error", (e as Error).message);
      }
    })();
    return () => { cancelled = true; closeStream(); };
  }, [downloadId, refresh]);

  const handleEdit = (cfg: LaunchConfig) => {
    // Mantém o id no initial → o save atualiza esta mesma entrada.
    setEditorInitial(cfg);
    setEditorOpen(true);
  };

  const handleNew = () => {
    setEditorInitial(null);
    setEditorOpen(true);
  };

  // Duplicar = abrir editor com uma cópia SEM id → o save gera um id novo e
  // cria outra entrada, deixando a original intacta.
  const handleDuplicate = (cfg: LaunchConfig) => {
    const { id: _id, ...rest } = cfg;
    setEditorInitial(rest);
    setEditorOpen(true);
  };

  const handleDeleteConfig = async (cfg: LaunchConfig) => {
    if (!cfg.id) return;
    const ok = window.confirm(
      `Remover a config de "${cfg.model}" (${cfg.backend})?\nNão dá pra desfazer.`,
    );
    if (!ok) return;
    await api.deleteConfig(cfg.id);
    await refresh();
  };

  // Identidade é o `id`: editar (mesmo trocando model/backend) atualiza a mesma
  // entrada, e salvar sem id cria uma nova — sem nunca sobrescrever outra config.
  const handleSave = async (cfg: LaunchConfig) => {
    await api.saveConfig(cfg);
    await refresh();
  };

  const handleLaunch = async (cfg: LaunchConfig) => {
    // Salva primeiro pra obter o id atribuído pelo backend, e lança ESSA config
    // (com id) — assim o launch não cria uma 2ª entrada e o auto-degrade
    // atualiza a config certa.
    try {
      const { config } = await api.saveConfig(cfg);
      const r = await api.launch(config);
      setLocalLaunch({ id: r.launch_id, config, active: true, origin: "owned" });
      setLaunchModalOpen(true);
      setEditorOpen(false);
    } catch (e) {
      // Ex.: 409 — já existe um launch ativo (um por vez).
      setError((e as Error).message);
    }
    await refresh();
  };

  // Launch múltiplo: modo router do llama-server — N configs sobem juntas,
  // cada uma num processo filho; o client escolhe pelo campo "model".
  const handleLaunchRouter = async (ids: string[]) => {
    try {
      const r = await api.launchRouter(ids);
      setLocalLaunch({ id: r.launch_id, config: r.config, active: true, origin: "owned" });
      setLaunchModalOpen(true);
    } catch (e) {
      setError((e as Error).message);
    }
    await refresh();
  };

  const handleStopLaunch = async () => {
    if (!localLaunch) return;
    await api.cancelLaunch(localLaunch.id);
    // Explicit dismissal is allowed to discard this session. Natural terminal
    // events use handleLaunchTerminal below and retain the modal history.
    setLocalLaunch(null);
    setLaunchModalOpen(false);
    await refresh();
  };

  // Restart 'soft': mantém a sessão (mesmo launchId/porta), só pede ao backend
  // pra matar e ressubir o llama-server com a mesma config. O cliente que
  // consome a porta fixa reconecta sozinho. Não fecha o modal nem limpa estado.
  const handleRestartLaunch = async () => {
    if (!localLaunch) return;
    try {
      await api.restartLaunch(localLaunch.id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleOpenLaunchLogs = () => {
    if (localLaunch) setLaunchModalOpen(true);
  };

  const handleLaunchTerminal = useCallback((id: string) => {
    setLocalLaunch((current) =>
      current && current.id === id ? { ...current, active: false } : current,
    );
  }, []);

  const handleStartDownload = async (req: {
    repo_id: string; revision: HfRevision; rel_paths: string[]; expected_files: HfExpectedFile[];
    subdir?: string | null; base_dir: string;
    force?: boolean;
  }) => {
    const generation = ++downloadGeneration.current;
    setDownloadEvents([]);
    setDownloadProgress({});
    setDownloadTerminal(null);
    setDownloadCancelling(false);
    setDownloadError(null);
    setDownloadPlan(null);
    setDownloadId(null);
    try {
      const r = await api.hfDownload(req);
      if (generation !== downloadGeneration.current) return;
      setDownloadPlan(r.plan);
      setDownloadId(r.download_id);
    } catch (e) {
      if (generation !== downloadGeneration.current) return;
      setDownloadError((e as Error).message);
      setDownloadTerminal("error");
    }
  };

  // "Atualizar": reusa o pipeline de download pros arquivos que divergiram.
  // rel_paths vem do detector (caminho remoto real, casa até repo com subdir);
  // root/subdir também vêm de lá e reconstroem exatamente a pasta atual do
  // modelo — o endpoint monta `<root>/<owner>/<repo>/<subdir>` e só aceita raiz
  // cadastrada, por isso não mandamos a pasta final. Re-baixa por cima: o
  // downloader vê o tamanho errado, apaga e refaz. Leva pra aba Download pra
  // acompanhar.
  const handleUpdateModel = async (u: ModelUpdate) => {
    if (!u.repo_id || u.rel_paths.length === 0) return;
    setTab("download");
    if (!u.root) {
      handleResetDownload();
      setDownloadError(
        `Não dá pra atualizar ${u.path}: o modelo não está sob nenhuma pasta ` +
        "cadastrada no layout owner/repo. Baixe novamente pela busca.",
      );
      setDownloadTerminal("error");
      return;
    }
    if (!u.download_revision || !FULL_SHA256.test(u.download_revision)) {
      handleResetDownload();
      setDownloadError(
        `Não dá pra atualizar ${u.path}: falta uma revisão imutável resolvida ` +
        "para este modelo. Baixe novamente pela busca.",
      );
      setDownloadTerminal("error");
      return;
    }

    const expected_files: HfExpectedFile[] = [];
    for (const rel of u.rel_paths) {
      const file = u.files.find((candidate) => candidate.rel === rel);
      if (!file || file.size <= 0 || !FULL_SHA256.test(file.oid)) {
        handleResetDownload();
        setDownloadError(`Não dá pra atualizar ${u.path}: falta metadado íntegro para ${rel}.`);
        setDownloadTerminal("error");
        return;
      }
      expected_files.push({ rel, expected_size: file.size, expected_oid: file.oid });
    }

    await handleStartDownload({
      repo_id: u.repo_id,
      revision: u.download_revision,
      rel_paths: u.rel_paths,
      expected_files,
      subdir: u.subdir,
      base_dir: u.root,
      // Sem force o downloader pularia por "já existe": a divergência mais
      // comum é sha256 com o MESMO tamanho, e o update não trocaria nada.
      force: true,
    });
  };

  const handleCancelDownload = async () => {
    if (!downloadId || downloadTerminal !== null || downloadCancelling) return;
    setDownloadCancelling(true);
    try {
      // A successful POST only asks the backend to cancel. The terminal state
      // still comes from SSE after the worker has joined.
      await api.cancelDownload(downloadId);
    } catch (e) {
      setDownloadCancelling(false);
      setDownloadError((e as Error).message);
    }
  };

  const handleResetDownload = () => {
    ++downloadGeneration.current;
    setDownloadId(null);
    setDownloadPlan(null);
    setDownloadEvents([]);
    setDownloadProgress({});
    setDownloadTerminal(null);
    setDownloadCancelling(false);
    setDownloadError(null);
  };

  // Spinner na aba Download quando há download em curso — sinaliza que dá
  // pra ficar em outra aba sem perder o progresso.
  const downloadActive = downloadId !== null && downloadTerminal === null;
  const tabs = [
    { id: "configs" as TabId,  label: "Configs",  icon: <Settings className="w-4 h-4" />, badge: configs.length },
    { id: "models"  as TabId,  label: "Models",   icon: <HardDrive className="w-4 h-4" />, badge: models.length },
    {
      id: "download" as TabId,
      label: "Download",
      icon: downloadActive
        ? <Loader2 className="w-4 h-4 animate-spin text-accent-fg" />
        : <Download className="w-4 h-4" />,
    },
    ...(mcpEnabled
      ? [{ id: "mcp" as TabId, label: "MCP", icon: <Plug className="w-4 h-4" /> }]
      : []),
    { id: "amd"     as TabId,  label: "AMD GPU",  icon: <Cpu className="w-4 h-4" /> },
    { id: "settings" as TabId, label: "Settings", icon: <SlidersHorizontal className="w-4 h-4" /> },
  ];

  return (
    <div className="min-h-screen bg-ink-950 text-ink-100">
      <Header system={system} backends={backends} onRefresh={refresh} />
      <Tabs tabs={tabs} active={tab} onChange={setTab} />

      <main className="px-6 py-5">
        {loading && (
          <div className="text-ink-400 text-sm">conectando ao backend Python…</div>
        )}
        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-200 rounded p-3 text-sm mono">
            ❌ {error}
            <div className="mt-1 text-red-300/70 text-xs">
              Verifique se Python + FastAPI/uvicorn estão instalados e no PATH.
            </div>
          </div>
        )}

        {!loading && !error && (
          <>
            {tab === "configs" && (
              <ConfigGrid
                configs={configs}
                models={models}
                estimates={estimates}
                onEdit={handleEdit}
                onDuplicate={handleDuplicate}
                onDelete={handleDeleteConfig}
                onLaunch={handleLaunch}
                onLaunchRouter={handleLaunchRouter}
                onNew={handleNew}
                activeLaunchCfg={localLaunch?.active ? localLaunch.config : null}
                launchActive={localLaunch?.active === true}
                onStopLaunch={handleStopLaunch}
                onRestartLaunch={handleRestartLaunch}
                onOpenLaunchLogs={handleOpenLaunchLogs}
              />
            )}
            {tab === "models"  && (
              <ModelsPage
                models={models}
                updates={updates}
                updatesChecking={updatesChecking}
                onAfterDelete={refresh}
                onRecheck={handleRecheckModel}
                onUpdate={handleUpdateModel}
              />
            )}
            {tab === "download" && (
              <DownloadPage
                downloadId={downloadId}
                plan={downloadPlan}
                events={downloadEvents}
                progress={downloadProgress}
                terminal={downloadTerminal}
                error={downloadError}
                cancelling={downloadCancelling}
                models={models}
                onStart={handleStartDownload}
                onCancel={handleCancelDownload}
                onReset={handleResetDownload}
              />
            )}
            {tab === "mcp" && mcpEnabled && <MCPPage />}
            {tab === "amd"     && <AmdPage />}
            {tab === "settings" && <SettingsPage onSaved={refresh} />}
          </>
        )}
      </main>

      <ConfigEditor
        open={editorOpen}
        initial={editorInitial}
        models={models}
        backends={backends}
        options={options}
        onClose={() => setEditorOpen(false)}
        onSave={handleSave}
        onLaunch={handleLaunch}
      />

      <LaunchModal
        open={launchModalOpen}
        launchId={localLaunch?.id ?? null}
        config={localLaunch?.config ?? null}
        onClose={() => {
          if (localLaunch && !localLaunch.active) setLocalLaunch(null);
          setLaunchModalOpen(false);
        }}
        onCancel={handleStopLaunch}
        onTerminal={handleLaunchTerminal}
      />
    </div>
  );
}
