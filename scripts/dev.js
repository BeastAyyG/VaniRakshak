#!/usr/bin/env node
/**
 * VaniRakshak — one-command integrated dev launcher.
 *
 * Boots the full vertical slice (FastAPI forensic backend on :8000 + Next.js
 * console on :3001) and prefixes every log line so you can tell them apart.
 * Ctrl-C tears both down.
 *
 *   node scripts/dev.js            # default: stub CM/ASV (no GPU/network needed)
 *   npm run dev:all
 *
 *   node scripts/dev.js --real     # real ECAPA-TDNN + WavLM AASIST detectors
 *   npm run dev:all:real
 *
 * --real installs vanirakshak-backend/requirements-asv.txt into the venv (one
 * extra `pip install`, first run only) and sets VR_ASV_BACKEND=ecapa,
 * VR_CM_BACKEND=aasist. Without it, every run uses the deterministic stub
 * detectors: no GPU, no model download, no network — see the "Notes" section
 * in the docs and docs/PROTOCOL.md before quoting stub-backend numbers as
 * real detection accuracy. Both pretrained models download from Hugging Face
 * on first use under --real (network access needed once).
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
const REAL = process.argv.includes("--real");

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
  const vpyPath = path.join(venvDir, IS_WIN ? "Scripts" : "bin", IS_WIN ? "python.exe" : "python");
  const venvExists = fs.existsSync(vpyPath);
  if (!venvExists) {
    console.log("\x1b[36m[dev]\x1b[0m creating backend virtualenv (first run only)...");
    const created = await run(python, ["-m", "venv", venvDir], { cwd: BACKEND });
    if (created !== 0) throw new Error("failed to create virtualenv");
    console.log("\x1b[36m[dev]\x1b[0m installing backend requirements...");
    const installed = await run(vpyPath, ["-m", "pip", "install", "-q", "-r", "requirements.txt"], { cwd: BACKEND });
    if (installed !== 0) throw new Error("failed to install backend requirements");
  }
  if (REAL) {
    // Marker file, not an import probe: importing torch/speechbrain just to
    // check presence is slow and can itself throw for unrelated reasons.
    // pip install is idempotent and cheap once already satisfied, so running
    // it every --real launch (not only on first venv creation) is simplest
    // and self-heals a venv that was created without --real before.
    console.log("\x1b[36m[dev]\x1b[0m --real: installing requirements-asv.txt (real ECAPA-TDNN + WavLM AASIST)...");
    const installed = await run(
      vpyPath,
      ["-m", "pip", "install", "-q", "-r", "requirements-asv.txt",
       "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
      { cwd: BACKEND },
    );
    if (installed !== 0) throw new Error("failed to install requirements-asv.txt for --real");
  }
  return vpyPath;
}

async function main() {
  const python = await ensureVenv(venvPython());

  console.log(`\x1b[36m[dev]\x1b[0m backend  : ${python} run.py  (${BASE})  [${REAL ? "REAL detectors" : "stub detectors"}]`);
  console.log(`\x1b[36m[dev]\x1b[0m frontend : next dev -p 3001  (http://localhost:3001)`);
  if (REAL) {
    console.log("\x1b[36m[dev]\x1b[0m --real: first run downloads pretrained weights from Hugging Face (needs network).");
  }

  const backendEnv = { ...process.env, VR_PORT: PORT, VR_HOST: "127.0.0.1" };
  if (REAL) {
    // An explicit VR_ASV_BACKEND/VR_CM_BACKEND the user already exported
    // wins — --real only fills in what they did not set themselves, same
    // convention as config.py's own real-backend preset.
    backendEnv.VR_ASV_BACKEND = backendEnv.VR_ASV_BACKEND || "ecapa";
    backendEnv.VR_CM_BACKEND = backendEnv.VR_CM_BACKEND || "aasist";
  }

  const backend = spawn(python, ["run.py"], {
    cwd: BACKEND,
    env: backendEnv,
    shell: IS_WIN,
  });
  backend.stdout.on("data", pipe("\x1b[35m[backend]\x1b[0m ", ""));
  backend.stderr.on("data", pipe("\x1b[35m[backend]\x1b[0m ", ""));

  let frontend = null;
  // --real loads two pretrained models onto CPU (and downloads them on first
  // run), which is well past the 60s default that's fine for the stub path.
  const healthy = await waitForHealth(REAL ? 240000 : 60000);
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
