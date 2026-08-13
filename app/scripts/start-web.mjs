import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = dirname(dirname(fileURLToPath(import.meta.url)));
// The backend binds loopback by default. For an explicit Tailscale deployment,
// pass a generic address, for example: LLM_LAUNCHER_HOST=100.x.y.z npm start.
const pythonFromVenv = process.platform === "win32"
  ? join(appDir, ".venv", "Scripts", "python.exe")
  : join(appDir, ".venv", "bin", "python");
const python = process.env.PYTHON || pythonFromVenv;

if (!existsSync(python)) {
  console.error(`Venv não encontrada em ${python}. Crie app/.venv antes de usar npm start.`);
  process.exit(1);
}

const child = spawn(python, [join(appDir, "api", "server.py")], {
  cwd: appDir,
  env: { ...process.env, PYTHONIOENCODING: "utf-8" },
  stdio: "inherit",
});

const forwardSignal = (signal) => child.kill(signal);
process.once("SIGINT", () => forwardSignal("SIGINT"));
process.once("SIGTERM", () => forwardSignal("SIGTERM"));
child.once("error", (error) => {
  console.error(`Falha ao iniciar backend web: ${error.message}`);
  process.exitCode = 1;
});
child.once("exit", (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
