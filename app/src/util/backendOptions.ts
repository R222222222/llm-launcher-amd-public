import type { BackendStatus, Option, Options } from "../api/types";

/**
 * The options endpoint is canonical; backend probing only tells us which of
 * those values may be sent. Keeping this in a pure helper makes it harder for
 * a backend-specific option to leak into the editor while capabilities are
 * changing.
 */
export function kvOptionsForBackend(
  options: Options,
  backend: BackendStatus | undefined,
  mode: "server" | "cli",
): Option<string>[] {
  if (!backend) return [];
  const supported = mode === "cli" ? backend.kv_types_cli : backend.kv_types_server;
  const allowed = new Set(supported);
  return options.kv_cache.filter((option) => allowed.has(option.value));
}

/** Pick the requested value, or the least surprising supported fallback. */
export function supportedKvValue(
  current: string,
  supported: Option<string>[],
): string | null {
  if (supported.some((option) => option.value === current)) return current;
  for (const preferred of ["q8_0", "f16"]) {
    if (supported.some((option) => option.value === preferred)) return preferred;
  }
  return supported[0]?.value ?? null;
}
