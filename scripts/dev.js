#!/usr/bin/env node
/**
 * VaniRakshak — one-command integrated dev launcher.
 *
 * Boots the full vertical slice (FastAPI forensic backend on :8000 + Next.js
 * console on :3001) and prefixes every log line so you can tell them apart.
 * Ctrl-C tears both down.
 *
 *   node scripts/dev.js
 *   npm run dev:all
 *
 * The backend interpreter is picked in this order:
 *   1. vanirakshak-backend/.venv            (preferred, isolated)
 *   2. VR_PYTHON env var
 *   3. `python3` / `python` on PATH
 *
 * If the venv does not exist yet it is created and `requirements.txt` is
 * installed into it, so a fresh clone works with a single command.
 */

const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const http = require("http");

const ROOT = path.resolve(__dirname, "..");
const BACKEND = path.join(ROOT, "vanirakshak-backend");
const FRONTEND = path.join(ROOT, "frontend");
const IS_WIN = process.platform === "win32";

const PORT = process.env.VR_PORT || "8000";
const BASE = `http://127.0.0.1:${PORT}`;

function venvPython() {
  const exe = IS_WIN ? "python.exe" : "python";
  const candidates = [
    path.join(BACKEND, ".venv", IS_WIN ? "Scripts" : "bin", exe),
    process.env.VR_PYTHON,
    IS_WIN ? "python" : "python3",
    "python",
  ].filter(Boolean);
  return candidates.find((c) => c.includes(path.sep) || c.includes("/") ? fs.existsSync(c) : true);
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { stdio: "inherit", shell: IS_WIN, ...opts });
    p.on("exit", (code) => resolve(code ?? 0));
  });
}

function pipe(prefix, color) {
  return (data) => {
    const lines = data.toString().split(/\r?\n/);
    for (const line of lines) {
      if (line.trim() === "") continue;
      process.stdout.write(`${color}${prefix}\x1b[0m ${line}\n`);
    }
  };
}

async function waitForHealth(timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const ok = await new Promise((resolve) => {
      const req = http.get(`${BASE}/api/v1/health`, (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      });
      req.on("error", () => resolve(false));
      req.setTimeout(1000, () => {
        req.destroy();
        resolve(false);
      });
    });
    if (ok) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

async function ensureVenv(python) {
  const venvDir = path.join(BACKEND, ".venv");
  if (fs.existsSync(path.join(venvDir, IS_WIN ? "Scripts" : "bin", IS_WIN ? "python.exe" : "python"))) {
    return path.join(venvDir, IS_WIN ? "Scripts" : "bin", IS_WIN ? "python.exe" : "python");
  }
  console.log("\x1b[36m[dev]\x1b[0m creating backend virtualenv (first run only)...");
  const created = await run(python, ["-m", "venv", venvDir], { cwd: BACKEND });
  if (created !== 0) throw new Error("failed to create virtualenv");
  const vpy = path.join(venvDir, IS_WIN ? "Scripts" : "bin", IS_WIN ? "python.exe" : "python");
  console.log("\x1b[36m[dev]\x1b[0m installing backend requirements...");
  const installed = await run(vpy, ["-m", "pip", "install", "-q", "-r", "requirements.txt"], { cwd: BACKEND });
  if (installed !== 0) throw new Error("failed to install backend requirements");
  return vpy;
}

async function main() {
  const python = await ensureVenv(venvPython());

  console.log(`\x1b[36m[dev]\x1b[0m backend  : ${python} run.py  (${BASE})`);
  console.log(`\x1b[36m[dev]\x1b[0m frontend : next dev -p 3001  (http://localhost:3001)`);

  const backend = spawn(python, ["run.py"], {
    cwd: BACKEND,
    env: { ...process.env, VR_PORT: PORT, VR_HOST: "127.0.0.1" },
    shell: IS_WIN,
  });
  backend.stdout.on("data", pipe("\x1b[35m[backend]\x1b[0m ", ""));
  backend.stderr.on("data", pipe("\x1b[35m[backend]\x1b[0m ", ""));

  let frontend = null;
  const healthy = await waitForHealth();
  if (healthy) {
    console.log("\x1b[36m[dev]\x1b[0m forensic engine online — starting dashboard");
  } else {
    console.log("\x1b[33m[dev]\x1b[0m backend health check timed out; starting dashboard anyway (it will show 'Engine offline')");
  }

  frontend = spawn(IS_WIN ? "npm.cmd" : "npm", ["run", "dev"], {
    cwd: FRONTEND,
    env: {
      ...process.env,
      NEXT_PUBLIC_BACKEND_URL: BASE,
      NEXT_PUBLIC_BACKEND_WS: `ws://127.0.0.1:${PORT}/ws/stream`,
    },
    shell: IS_WIN,
  });
  frontend.stdout.on("data", pipe("\x1b[32m[frontend]\x1b[0m ", ""));
  frontend.stderr.on("data", pipe("\x1b[32m[frontend]\x1b[0m ", ""));

  const shutdown = () => {
    console.log("\n\x1b[36m[dev]\x1b[0m shutting down...");
    if (frontend) frontend.kill("SIGTERM");
    backend.kill("SIGTERM");
    setTimeout(() => process.exit(0), 400);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
  backend.on("exit", (code) => {
    console.log(`\x1b[35m[backend]\x1b[0m exited (${code})`);
    if (frontend) frontend.kill("SIGTERM");
    process.exit(code ?? 0);
  });
}

main().catch((e) => {
  console.error(`\x1b[31m[dev]\x1b[0m ${e.message}`);
  process.exit(1);
});
