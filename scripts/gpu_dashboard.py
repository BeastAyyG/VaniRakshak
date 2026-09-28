"""Read-only dashboard for scripts/gpu_pipeline.py.

Shows each stage's status and progress, the tail of its log, live GPU
utilisation / memory / temperature (nvidia-smi), and the headline results
(T1 grade, P1-T, STOP-C). It has no controls: it can't start, stop or change
anything, so exposing it to your own devices is low-risk.

Binds to 127.0.0.1 by default. Add ``--host <tailscale-ip>`` (repeatable) to
reach it from your other devices on the tailnet, without opening it to the LAN.

    python scripts/gpu_dashboard.py --host 127.0.0.1 --host 100.x.y.z --port 8787
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

OUT = Path(os.environ.get("VR_PIPELINE_OUT", "C:/vr-out"))
ROOT = OUT / "pipeline"
RESULT_FILES = {
    "t1_grade": OUT / "t1" / "t1_grade.json",
    "telephony": OUT / "d2" / "telephony_stopc.json",
    "hindi_asr": OUT / "w2c" / "hindi_asr.json",
}


def _gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        name, util, used, total, temp, power = [x.strip() for x in out.split(",")]
        return {"name": name, "util": float(util), "mem_used": float(used), "mem_total": float(total),
                "temp": float(temp), "power": float(power) if power not in ("[N/A]", "") else None}
    except Exception as e:
        return {"error": f"{type(e).__name__}"}


def _tail(path: Path, n: int = 25) -> list:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def snapshot() -> dict:
    # Every runner writes its own state*.json; merge them for one view.
    state: dict = {"stages": {}, "pid": None}
    for f in sorted(ROOT.glob("state*.json")):
        try:
            part = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        state["stages"].update(part.get("stages", {}))
        state["pid"] = state["pid"] or part.get("pid")
    for name, st in state.get("stages", {}).items():
        st["log_tail"] = _tail(ROOT / f"{name}.log")
    results = {}
    for k, p in RESULT_FILES.items():
        try:
            results[k] = json.loads(p.read_text())
        except (OSError, ValueError):
            pass
    return {"now": time.strftime("%Y-%m-%d %H:%M:%S"), "state": state, "gpu": _gpu(), "results": results}


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VaniRakshak GPU Pipeline</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--ok:#1f7a4d;--bad:#b3261e;--run:#1f5fb3;--bar:#1f5fb3}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--muted:#9d9c95;--line:#33332f;--ok:#5cc28f;--bad:#f2867d;--run:#7fb0f5;--bar:#7fb0f5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.k{color:var(--muted);font-size:12px}.v{font-size:20px;font-variant-numeric:tabular-nums}
.stage{margin-bottom:10px}.row{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap}
.pill{font-size:12px;padding:2px 8px;border-radius:999px;border:1px solid currentColor}
.done{color:var(--ok)}.failed{color:var(--bad)}.running{color:var(--run)}.pending{color:var(--muted)}
.bar{height:6px;background:var(--line);border-radius:3px;margin:8px 0 4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--bar)}
details summary{cursor:pointer;color:var(--muted);font-size:12px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,Consolas,monospace;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;max-height:260px;overflow:auto;margin:6px 0 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;overflow-wrap:anywhere}
.res{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
</style></head><body><main>
<h1>VaniRakshak GPU pipeline</h1><div class="sub" id="sub">loading…</div>
<div class="grid" id="gpu"></div>
<h2>Stages</h2><div id="stages"></div>
<h2 style="margin-top:18px">Results</h2><div class="res" id="res"></div>
</main><script>
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const yes=b=>b===true?'<span class="done">PASS</span>':b===false?'<span class="failed">FAIL</span>':'—';
function card(k,v){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`}
async function tick(){
 let d;try{d=await (await fetch('api/state',{cache:'no-store'})).json()}catch(e){document.getElementById('sub').textContent='dashboard server unreachable';return}
 const g=d.gpu;document.getElementById('sub').textContent=`${g.name||'GPU'} · updated ${d.now} · pipeline ${d.state.pid?'running (pid '+d.state.pid+')':'idle'}`;
 document.getElementById('gpu').innerHTML=g.error?card('GPU',esc(g.error)):card('GPU util',g.util+' %')+card('VRAM',(g.mem_used/1024).toFixed(1)+' / '+(g.mem_total/1024).toFixed(1)+' GB')+card('Temp',g.temp+' °C')+card('Power',g.power==null?'—':g.power.toFixed(0)+' W');
 const st=d.state.stages||{};document.getElementById('stages').innerHTML=Object.entries(st).map(([n,s])=>{
  const p=s.progress||{};const pct=p.total?Math.round(100*p.done/p.total):(s.status==='done'?100:0);
  return `<div class="card stage"><div class="row"><b>${esc(n)}</b><span class="pill ${esc(s.status)}">${esc(s.status)}</span></div>
  <div class="bar"><i style="width:${pct}%"></i></div><div class="row k"><span>${esc(s.message||s.error||'')}</span><span>${p.total?p.done+'/'+p.total:''}</span></div>
  <details><summary>log</summary><pre>${esc((s.log_tail||[]).join('\n'))}</pre></details></div>`}).join('')||'<div class="card">no pipeline state yet</div>';
 const r=d.results,out=[];
 if(r.t1_grade){const t=r.t1_grade;out.push(`<div class="card"><h2>T1 — decision layer (gate split)</h2><table>
  <tr><td>P1-S (≥ +30 pp at worst cell)</td><td>${yes(t.P1_S_pass)} ${t.p1s_rise_at_worst_cell_pp} pp</td></tr>
  <tr><td>P1-F (≤ +5 pp challenge @60 s)</td><td>${yes(t.P1_F_pass)} max ${t.p1f_max_rise_pp} pp</td></tr>
  <tr><td>Worst cell</td><td>${esc(t.worst_cell)}</td></tr>
  <tr><td>Genuine BLOCK / call (not pre-reg.)</td><td>${(100*t.not_pre_registered.genuine_block_by_end_after_max).toFixed(1)} % max</td></tr></table></div>`)}
 for(const dev of ['t2_cuda','t2_cpu']){const s=(st[dev]||{}).result;if(s)out.push(`<div class="card"><h2>P1-T — ${dev==='t2_cuda'?'CUDA fp16':'CPU fp32'}</h2><table>
  <tr><td>Backends</td><td>${esc(s.backends.asv)} / ${esc(s.backends.cm)}</td></tr><tr><td>Checks</td><td>${s.checks_passed}/${s.checks_total}</td></tr>
  ${Object.entries(s.p1t).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${yes(v)}</td></tr>`).join('')}<tr><td><b>P1-T</b></td><td>${yes(s.P1_T_pass)}</td></tr></table></div>`)}
 if(r.telephony){const t=r.telephony;out.push(`<div class="card"><h2>STOP-C — telephony (incumbent)</h2><table><tr><th>Channel</th><th>FRR</th><th>Worst miss</th></tr>
  ${Object.entries(t.conditions).map(([c,v])=>`<tr><td>${esc(c)}</td><td>${v.bonafide_frr_pct} %</td><td>${v.worst_miss_pct} % (${esc(v.worst_attack)})</td></tr>`).join('')}
  <tr><td colspan="2"><b>STOP-C triggered</b></td><td>${t.stop_c.triggered==null?'—':t.stop_c.triggered?'<span class="failed">YES</span>':'<span class="done">no</span>'}</td></tr></table></div>`)}
 if(r.hindi_asr){const h=r.hindi_asr;out.push(`<div class="card"><h2>Hindi challenge ASR</h2><table><tr><th>Engine</th><th>Hindi pass</th><th>English pass</th></tr>
  ${Object.entries(h.engines||{}).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${v.hi_pass_pct} %</td><td>${v.en_pass_pct} %</td></tr>`).join('')}</table></div>`)}
 document.getElementById('res').innerHTML=out.join('')||'<div class="card">no results yet</div>';
}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/state"):
            body, ctype = json.dumps(snapshot()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", action="append", default=None)
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    hosts = args.host or ["127.0.0.1"]
    servers = [ThreadingHTTPServer((h, args.port), Handler) for h in hosts]
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    for h in hosts:
        print(f"dashboard: http://{h}:{args.port}/", flush=True)
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
