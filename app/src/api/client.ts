// Cliente HTTP do backend Python — wrappers tipados sobre fetch.
//
// Em produção a API é sempre same-origin. VITE_API_BASE permanece disponível
// somente para desenvolvimento fora do FastAPI; o proxy do Vite é o caminho
// padrão para dev.

import type {
  ActiveLaunch,
  AppSettings,
  BackendStatus,
  ConfigurableBackend,
  DeletePlan,
  DeleteResult,
  Estimate,
  GgufMeta,
  HfDownloadPlan,
  HfExpectedFile,
  HfResolvedRef,
  HfRevision,
  HfRepoListing,
  HfSearchResult,
  LaunchConfig,
  LaunchEventCursor,
  LmsLoadResult,
  LmsStatus,
  McpServer,
  McpStatus,
  ModelDefaults,
  ModelInfo,
  ModelSampling,
  ModelUpdate,
  AmdStatus,
  Options,
  RouterLaunchConfig,
  SystemInfo,
} from "./types";

let _baseUrl: string | null = null;

export async function apiBase(): Promise<string> {
  if (_baseUrl) return _baseUrl;
  const configuredBase = import.meta.env.DEV ? import.meta.env.VITE_API_BASE : undefined;
  const baseUrl = configuredBase || window.location.origin;
  _baseUrl = baseUrl;
  return baseUrl;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const base = await apiBase();
  const r = await fetch(`${base}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      detail = body?.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`API ${path}: ${detail}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  options:    () => jsonFetch<Options>("/api/options"),
  backends:   () => jsonFetch<BackendStatus[]>("/api/backends"),
  system:     () => jsonFetch<SystemInfo>("/api/system"),
  models:     () => jsonFetch<ModelInfo[]>("/api/models"),
  modelMeta:  (model: string) =>
    jsonFetch<GgufMeta>("/api/models/meta", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  modelContextOptions: (model: string) =>
    jsonFetch<{ options: { value: number; label: string }[] }>("/api/models/context-options", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  // Samplers que o modelo receberia agora + a procedência deles.
  modelSampling: (model: string, config?: LaunchConfig) =>
    jsonFetch<ModelSampling>("/api/models/sampling", {
      method: "POST",
      body: JSON.stringify({ model, config: config ?? null }),
    }),
  // Config inteira derivada do modelo + VRAM da máquina.
  modelDefaults: (model: string, backend: string, mode = "server") =>
    jsonFetch<ModelDefaults>("/api/models/defaults", {
      method: "POST",
      body: JSON.stringify({ model, backend, mode }),
    }),

  // Checagem de atualização de GGUFs. Sem deep: barato (tamanho + oid
  // registrado/cacheado), pra rodar na inicialização. Deep: hasheia o arquivo
  // pra confirmar — botão manual por modelo.
  modelsUpdates: (deep = false) =>
    jsonFetch<{ results: ModelUpdate[] }>(`/api/models/updates?deep=${deep ? "true" : "false"}`),
  checkModelUpdate: (model: string, deep = true) =>
    jsonFetch<ModelUpdate>("/api/models/check-update", {
      method: "POST",
      body: JSON.stringify({ model, deep }),
    }),

  listConfigs: () => jsonFetch<LaunchConfig[]>("/api/configs"),
  saveConfig:  (cfg: LaunchConfig) =>
    jsonFetch<{ ok: boolean; config: LaunchConfig }>("/api/configs", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  deleteConfig: (id: string) =>
    jsonFetch<{ ok: boolean; removed: number }>("/api/configs", {
      method: "DELETE",
      body: JSON.stringify({ id }),
    }),

  estimateMany: (items: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    n_cpu_moe: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }[]) =>
    jsonFetch<{ ok: boolean; estimate?: Estimate; error?: string; model: string; backend: string }[]>(
      "/api/estimate-many",
      { method: "POST", body: JSON.stringify({ items }) },
    ),
  estimate: (req: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    n_cpu_moe: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }) =>
    jsonFetch<Estimate>("/api/estimate", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  buildCommand: (cfg: LaunchConfig) =>
    jsonFetch<{ command: string; mode: string }>("/api/build-command", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  suggestNCpuMoe: (req: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }) =>
    jsonFetch<{ n_cpu_moe: number }>("/api/suggest/n-cpu-moe", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  suggestNGpuLayers: (req: {
    model: string;
    backend: string;
    context_window: number;
    kv_cache: string;
    parallel_slots: number;
    gpu_layers: number;
    mmproj: string | null;
    cache_ram: number;
    mode: "server" | "cli";
  }) =>
    jsonFetch<{ n_gpu_layers: number }>("/api/suggest/n-gpu-layers", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  launch: (cfg: LaunchConfig) =>
    jsonFetch<{ launch_id: string }>("/api/launch", {
      method: "POST",
      body: JSON.stringify(cfg),
    }),
  // Modo router: sobe N configs de uma vez (llama-server --models-preset).
  launchRouter: (ids: string[]) =>
    jsonFetch<{ launch_id: string; config: RouterLaunchConfig }>("/api/launch-router", {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  cancelLaunch: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/launch/${id}/cancel`, { method: "POST" }),
  restartLaunch: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/launch/${id}/restart`, { method: "POST" }),
  listLaunches: () =>
    jsonFetch<ActiveLaunch[]>("/api/launches"),

  // ─── HuggingFace ─────────────────────────────────────────────────────
  hfResolve: (url: string) =>
    jsonFetch<HfResolvedRef>("/api/hf/resolve", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),
  hfList: (req: { repo_id: string; revision: string }) =>
    jsonFetch<HfRepoListing>("/api/hf/list", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  hfSearch: (query: string, limit = 25) =>
    jsonFetch<{ results: HfSearchResult[] }>("/api/hf/search", {
      method: "POST",
      body: JSON.stringify({ query, limit }),
    }),
  hfDownload: (req: {
    repo_id: string;
    revision: HfRevision;
    rel_paths: string[];
    expected_files: HfExpectedFile[];
    subdir?: string | null;
    base_dir: string;
    // "Atualizar": re-baixa por cima mesmo com tamanho igual (sha256 divergiu).
    force?: boolean;
  }) =>
    jsonFetch<{ download_id: string; plan: HfDownloadPlan }>("/api/hf/download", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  cancelDownload: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/hf/download/${id}/cancel`, { method: "POST" }),

  // ─── delete model ────────────────────────────────────────────────────
  planDeleteModel: (model: string) =>
    jsonFetch<DeletePlan>("/api/models/plan-delete", {
      method: "POST",
      body: JSON.stringify({ model, confirm: false }),
    }),
  deleteModel: (model: string) =>
    jsonFetch<DeleteResult>("/api/models", {
      method: "DELETE",
      body: JSON.stringify({ model, confirm: true }),
    }),

  // ─── LM Studio ───────────────────────────────────────────────────────
  lmsStatus: () => jsonFetch<LmsStatus>("/api/lms/status"),
  lmsLoad: (req: { model: string; context_window: number; parallel_slots: number }) =>
    jsonFetch<LmsLoadResult>("/api/lms/load", {
      method: "POST",
      body: JSON.stringify(req),
    }),

  // ─── AMD GPU telemetry ─────────────────────────────────────────────────
  gpu: () => jsonFetch<AmdStatus>("/api/gpu"),

  // ─── MCP servers ─────────────────────────────────────────────────────
  mcpList: () => jsonFetch<McpServer[]>("/api/mcp"),
  mcpSave: (s: { id?: string; name: string; cwd: string; command: string; enabled?: boolean }) =>
    jsonFetch<{ ok: boolean; server: McpServer; status: McpStatus }>("/api/mcp", {
      method: "POST",
      body: JSON.stringify(s),
    }),
  mcpDelete: (id: string) =>
    jsonFetch<{ ok: boolean }>(`/api/mcp/${id}`, { method: "DELETE" }),
  mcpStart: (id: string) =>
    jsonFetch<McpStatus>(`/api/mcp/${id}/start`, { method: "POST" }),
  mcpStop: (id: string) =>
    jsonFetch<McpStatus>(`/api/mcp/${id}/stop`, { method: "POST" }),
  mcpLogs: (id: string, limit = 500) =>
    jsonFetch<{ logs: { ts: number; kind: string; text: string }[] }>(
      `/api/mcp/${id}/logs?limit=${limit}`,
    ),

  // ─── app settings ────────────────────────────────────────────────────
  getSettings: () => jsonFetch<AppSettings>("/api/settings"),
  // Só envia os campos editáveis — backend_paths_defaults é read-only.
  saveSettings: (settings: {
    model_paths: string[];
    backend_paths: Partial<Record<ConfigurableBackend, string>>;
  }) =>
    jsonFetch<AppSettings>("/api/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),
};

export async function eventsStream(
  launchId: string,
  after: LaunchEventCursor = null,
): Promise<EventSource> {
  const base = await apiBase();
  const query = after === null ? "" : `?after=${encodeURIComponent(String(after))}`;
  return new EventSource(`${base}/api/launch/${launchId}/events${query}`);
}

export async function downloadEventsStream(downloadId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/hf/download/${downloadId}/events`);
}

export async function mcpEventsStream(serverId: string): Promise<EventSource> {
  const base = await apiBase();
  return new EventSource(`${base}/api/mcp/${serverId}/events`);
}
