"""Serve the research results dashboard.

This is the *results* view -- what the experiments concluded -- as distinct
from scripts/serve_dashboard.py, which is the *progress* view (which runs
are still going).

Everything is read from reports/ on each request, so the page stays current
as new artifacts land. The dose-response chart is rendered server-side as
SVG: fewer moving parts than client-side drawing, and it is testable in
Python.

Nothing here constructs a model or touches the GPU -- a dashboard that loads
a 95M-parameter frontend on every refresh would be unusable.

Usage:
    python scripts/serve_results_dashboard.py [--port 8792]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
REPORTS = REPO / "reports"
HTML = REPO / "results-dashboard.html"
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import importlib  # noqa: E402

import ood_comparison  # noqa: E402
from ood_comparison import REGISTRY, load  # noqa: E402


def fresh_registry():
    """Reload the checkpoint registry so newly measured checkpoints appear
    without restarting the server.

    REGISTRY is module-level, so a long-lived process keeps showing whatever
    existed at startup -- which is exactly how seed 44 stayed invisible after
    it was scored. Reloading is cheap (the module is tiny) and removes the
    need to bounce onto a new port every time a report lands.
    """
    importlib.reload(ood_comparison)
    return ood_comparison.REGISTRY, ood_comparison.load

SEEDS = (42, 43)


def newest(pattern: str) -> Path | None:
    hits = sorted(REPORTS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def read_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt report must not kill the page
        return {}


# --------------------------------------------------------------------------
# dose-response chart, rendered as SVG
# --------------------------------------------------------------------------

def dose_chart_svg(rows: list[dict]) -> str:
    """Line chart: EER vs augmentation dose, one series per seed.

    `rows` = [{dose, seed, eer, lo, hi}, ...]
    """
    if not rows:
        return "<p class='muted'>no dose-response data yet</p>"

    W, H = 900, 360
    ml, mr, mt, mb = 62, 24, 22, 54
    pw, ph = W - ml - mr, H - mt - mb

    eers = [r["eer"] for r in rows]
    los = [r["lo"] for r in rows]
    his = [r["hi"] for r in rows]
    y0, y1 = min(los), max(his)
    pad = max((y1 - y0) * 0.15, 1.0)
    y0, y1 = y0 - pad, y1 + pad
    x0, x1 = 0.0, 1.0

    def sx(d):
        return ml + (d - x0) / (x1 - x0) * pw

    def sy(v):
        return mt + ph - (v - y0) / (y1 - y0) * ph

    P = [f"<svg viewBox='0 0 {W} {H}' width='100%' "
         f"role='img' aria-label='EER by augmentation dose'>",
         f"<rect x='{ml}' y='{mt}' width='{pw}' height='{ph}' fill='#161b22' "
         f"stroke='#30363d' rx='6'/>"]

    # y gridlines
    steps = 5
    for i in range(steps + 1):
        v = y0 + (y1 - y0) * i / steps
        y = sy(v)
        P.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{ml+pw}' y2='{y:.1f}' "
                 f"stroke='#30363d' stroke-width='1'/>")
        P.append(f"<text x='{ml-10}' y='{y+4:.1f}' fill='#8b949e' "
                 f"font-size='12' text-anchor='end'>{v:.1f}</text>")

    # x ticks
    for d in (0.0, 0.25, 0.5, 1.0):
        x = sx(d)
        P.append(f"<line x1='{x:.1f}' y1='{mt}' x2='{x:.1f}' y2='{mt+ph}' "
                 f"stroke='#30363d' stroke-dasharray='3 3'/>")
        P.append(f"<text x='{x:.1f}' y='{mt+ph+22}' fill='#8b949e' "
                 f"font-size='12' text-anchor='middle'>{d}</text>")
    P.append(f"<text x='{ml+pw/2}' y='{H-12}' fill='#8b949e' font-size='13' "
             f"text-anchor='middle'>codec augmentation dose "
             f"(0 = off, 1 = today's setting)</text>")
    P.append(f"<text x='14' y='{mt+ph/2}' fill='#8b949e' font-size='13' "
             f"text-anchor='middle' "
             f"transform='rotate(-90 14 {mt+ph/2})'>ITW full_call EER %</text>")

    colors = {42: "#58a6ff", 43: "#d29922"}
    for seed in SEEDS:
        pts = sorted([r for r in rows if r["seed"] == seed],
                     key=lambda r: r["dose"])
        if not pts:
            continue
        c = colors.get(seed, "#8b949e")
        # CI band
        top = " ".join(f"{sx(r['dose']):.1f},{sy(r['hi']):.1f}" for r in pts)
        bot = " ".join(f"{sx(r['dose']):.1f},{sy(r['lo']):.1f}" for r in pts[::-1])
        P.append(f"<polygon points='{top} {bot}' fill='{c}' opacity='0.13'/>")
        # line + points
        line = " ".join(f"{sx(r['dose']):.1f},{sy(r['eer']):.1f}" for r in pts)
        P.append(f"<polyline points='{line}' fill='none' stroke='{c}' "
                 f"stroke-width='2.5'/>")
        for r in pts:
            x, y = sx(r["dose"]), sy(r["eer"])
            P.append(f"<line x1='{x:.1f}' y1='{sy(r['lo']):.1f}' "
                     f"x2='{x:.1f}' y2='{sy(r['hi']):.1f}' stroke='{c}' "
                     f"stroke-width='1.4' opacity='.8'/>")
            P.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{c}'/>")
            P.append(f"<text x='{x:.1f}' y='{y-12:.1f}' fill='{c}' "
                     f"font-size='11' text-anchor='middle'>{r['eer']:.1f}</text>")

    # legend
    lx, ly = ml + 12, mt + 12
    for seed in SEEDS:
        c = colors.get(seed, "#8b949e")
        P.append(f"<rect x='{lx}' y='{ly}' width='12' height='12' fill='{c}' "
                 f"rx='2'/>")
        P.append(f"<text x='{lx+18}' y='{ly+11}' fill='#e6edf3' "
                 f"font-size='12'>seed {seed}</text>")
        ly += 20

    P.append("</svg>")
    return "".join(P)


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------

def build_results() -> dict:
    d21 = read_json(newest("docs21_adjudication_*.json"))
    d20 = read_json(newest("docs20_adjudication_*.json"))
    g4 = read_json(newest("g4_adjudication_*.json"))
    spread = read_json(REPORTS / "seed_spread_by_metric_class.json")

    # dose curve
    rows = []
    table = d21.get("table") or {}
    for dose_s, seeds in table.items():
        try:
            dose = float(dose_s)
        except ValueError:
            continue
        for seed_s, val in seeds.items():
            if not val:
                continue
            rows.append({"dose": dose, "seed": int(seed_s),
                         "eer": val[0], "lo": val[1][0], "hi": val[1][1]})

    # out-of-domain across every checkpoint
    ood = []
    reg, load_ci = fresh_registry()
    for label, codec, _rb, c_name, f_name in reg:
        c, f = load_ci(c_name), load_ci(f_name)
        if not c and not f:
            continue
        ood.append({
            "model": label,
            "codec": codec,
            "clean": (c or {}).get("eer_pct"),
            "clean_ci": (c or {}).get("ci95_pct"),
            "full_call": (f or {}).get("eer_pct"),
            "full_call_ci": (f or {}).get("ci95_pct"),
        })
    ood.sort(key=lambda r: (r["full_call"] is None,
                            r["full_call"] if r["full_call"] is not None else 0))

    # deployment facts -- cheap checkpoint reads only, no model construction
    dep = {"ok": False, "detail": "not checked"}
    try:
        sys.path.insert(0, str(REPO / "vanirakshak-backend"))
        from vanirakshak.detectors import cm_aasist as cm

        import torch

        p = cm.weights_path()
        if p.exists():
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            dep = {
                "ok": True,
                "weights": str(p),
                "codec_augment": ck.get("codec_augment"),
                "calibrated": bool(ck.get("calibration")),
                "dev_eer": ck.get("dev_eer"),
                "default_backend": "aasist (auto-selected, weights present)",
            }
        else:
            dep = {"ok": False, "detail": f"weights missing: {p}"}
    except Exception as exc:  # noqa: BLE001
        dep = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}

    verdicts = [
        {"name": "G4 RawBoost", "status": g4.get("OVERALL", "—"),
         "note": "A19/A20 vs a 12% bar"},
        {"name": "docs/20 aug-off", "status": d20.get("OVERALL", "—"),
         "note": d20.get("criteria", {})
                    .get("primary_itw_full_call_better_off", {}).get("status", "")},
        {"name": "docs/21 dose", "status": d21.get("verdict", "—"),
         "note": d21.get("verdict_detail", "")},
        {"name": "Deployment", "status": "PASS" if dep.get("ok") else "CHECK",
         "note": "arm B wired as default"},
    ]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdicts": verdicts,
        "dose_rows": rows,
        "dose_chart_svg": dose_chart_svg(rows),
        "dose_monotonic": {
            str(k): v for k, v in (d21.get("monotonic") or {}).items()},
        "ood": ood,
        "seed_spread": spread,
        "deployment": dep,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        # The page may be opened from a different origin -- e.g. an IDE
        # preview proxy serving the file on its own port -- in which case its
        # relative fetch("/api/results") misses and it falls back to this
        # absolute origin. That is a cross-origin request, so it needs CORS.
        # Localhost-only dashboard; no credentials are involved.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/results"):
            self._send(json.dumps(build_results()).encode("utf-8"),
                       "application/json")
            return
        try:
            page = HTML.read_bytes()
        except OSError:
            self._send(b"results-dashboard.html not found", "text/plain")
            return
        self._send(page, "text/html; charset=utf-8")

    def log_message(self, *args):  # noqa: A002 - keep the console quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    # Fail fast and loudly: if the payload raises, we want to see it now
    # rather than on the first request.
    try:
        build_results()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: build_results() raised at startup: "
              f"{type(exc).__name__}: {exc}")
    print(f"results dashboard: http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
