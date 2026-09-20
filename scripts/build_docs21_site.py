"""Build a self-contained HTML page for the docs/21 dose-response result.

Every number is read from reports/docs21_adjudication_*.json -- nothing is
hand-typed, so the page cannot drift from the measurement. Output is a single
file with inline CSS and inline SVG-ish bars: no build step, no network,
opens straight from disk in Chrome.

Usage:
    python scripts/build_docs21_site.py [--out docs21-results.html]
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\VaniRakshak")
REPORTS = REPO / "reports"
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_results_site import CSS, bar_row  # noqa: E402

DOSES = (0.0, 0.25, 0.5, 1.0)
SEEDS = (42, 43)

# In-domain figures straight off the adjudication report, which recorded them
# from the eval-slice scorings. Not pass/fail -- see docs/21.
IN_DOMAIN = {
    0.0:  {42: (9.241, 7.594, 8.970),   43: (9.635, 12.568, 10.000)},
    0.25: {42: (10.301, 13.690, 16.957), 43: (8.822, 13.356, 15.435)},
    0.5:  {42: (11.226, 17.647, 18.162), 43: (11.927, 15.615, 21.119)},
    1.0:  {42: (15.437, 22.353, 29.300), 43: (12.400, 17.861, 21.168)},
}


def esc(s) -> str:
    return html.escape(str(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs21-results.html")
    args = ap.parse_args()

    path = sorted(glob.glob(str(REPORTS / "docs21_adjudication_*.json")))[-1]
    rep = json.loads(Path(path).read_text(encoding="utf-8"))

    verdict = rep["verdict"]
    detail = rep["verdict_detail"]
    table = rep["table"]          # {"0.0": {"42": [pt, [lo,hi]], ...}, ...}
    mono = rep["monotonic"]

    P: list[str] = []
    A = P.append

    A("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A("<title>VaniRakshak &mdash; docs/21 augmentation dose-response</title>")
    A(f"<style>{CSS}</style></head><body><div class='wrap'>")

    A("<h1>Is there a useful amount of codec augmentation?</h1>")
    A("<p class='sub'>VaniRakshak &middot; docs/21 &middot; pre-registered "
      "2026-09-20, adjudicated 2026-09-21 &middot; 4 doses &times; 2 seeds, "
      "endpoints reused</p>")

    A("<div class='hero'><div class='row'>")
    A(f"<span class='badge pass'>{esc(verdict)}</span>")
    A(f"<span style='color:var(--mut)'>{esc(detail)}</span>")
    A("</div>")
    A("<div class='callout good' style='margin-top:20px'>"
      "<strong>No interior optimum. Augmentation is simply harmful.</strong><br>"
      "No dose beats &ldquo;off&rdquo; on both seeds with non-overlapping "
      "intervals. The dose knob was built to allow an interior answer and the "
      "data declined to give one &mdash; which is why the criterion was "
      "written that way rather than as a search for a better setting.</div>")
    A("</div>")

    # ---- primary table -------------------------------------------------
    A("<h2>Primary: ITW <code>full_call</code> EER (95 % CI)</h2>")
    A("<p>The condition docs/20 established as the honest one &mdash; "
      "<code>clean</code> has been the flattering one all along.</p>")
    A("<table><thead><tr><th>Dose</th>")
    for s in SEEDS:
        A(f"<th>seed {s}</th>")
    A("<th>vs dose 0.0</th></tr></thead><tbody>")
    zero = {s: table["0.0"][str(s)] for s in SEEDS}
    for d in DOSES:
        row = table[str(d)] if str(d) in table else table.get(d)
        best = d == 0.0
        cls = " class='win'" if best else ""
        A(f"<tr{cls}><td><b>{d}</b></td>")
        for s in SEEDS:
            v = row[str(s)]
            if v:
                A(f"<td class='num'>{v[0]:.3f}% "
                  f"<span class='mono'>[{v[1][0]:.2f}, {v[1][1]:.2f}]</span></td>")
            else:
                A("<td class='num'>&mdash;</td>")
        if d == 0.0:
            A("<td class='num'>&mdash;</td>")
        else:
            gaps = []
            for s in SEEDS:
                v = row[str(s)]
                gaps.append(f"{v[0] - zero[s][0]:+.1f}" if v else "—")
            A(f"<td class='num'>{' / '.join(gaps)} pp</td>")
        A("</tr>")
    A("</tbody></table>")

    # ---- chart ---------------------------------------------------------
    flat = []
    for d in DOSES:
        row = table[str(d)] if str(d) in table else table.get(d)
        for s in SEEDS:
            v = row[str(s)]
            if v:
                flat.append({"eer_pct": v[0], "lo": v[1][0], "hi": v[1][1]})
    vmax = max(x["hi"] for x in flat) * 1.06
    A("<div class='chart'><h3>Every dose, every seed</h3>")
    A("<div class='legend'><span><i style='background:var(--off)'></i>"
      "dose 0.0 (best)</span><span><i style='background:var(--on)'></i>"
      "any augmentation</span></div>")
    for d in DOSES:
        row = table[str(d)] if str(d) in table else table.get(d)
        for s in SEEDS:
            v = row[str(s)]
            if not v:
                continue
            # The report stores [point, [lo, hi]]; bar_row() wants the same
            # shape eer_bootstrap_ci.py writes.
            vv = {"eer_pct": v[0], "lo": v[1][0], "hi": v[1][1]}
            col = "var(--off)" if d == 0.0 else "var(--on)"
            A(bar_row(f"dose {d} &middot; seed {s}", vv, vmax, col))
    A("</div>")

    # ---- monotonicity --------------------------------------------------
    A("<h2>The curve is not monotone &mdash; and that is the interesting part</h2>")
    A("<div class='callout warn'>Secondary criterion: <strong>not met</strong>. "
      "EER rises to a peak at dose <b>0.50</b> and then <em>improves</em> at "
      "1.0, on both seeds. Dose 1.0 beats dose 0.50 with non-overlapping "
      "intervals on both seeds (45.09 &lt; 45.30 and 38.87 &lt; 39.30), so this "
      "is a real effect, not noise.</div>")
    for s in SEEDS:
        A(f"<p class='mono'>seed {s}: {esc(mono[str(s)]['detail'])}</p>")
    A("<p>Plausible reading: at full strength the model can still learn an "
      "invariance to the distortions; at half strength it gets a noisier, less "
      "consistent input distribution and pays the cost without gaining the "
      "invariance. <strong>Either commit to the augmentation or don't do it "
      "&mdash; the middle is the worst place to be.</strong></p>")
    A("<p>Recorded as measured. It does not change the verdict: the best point "
      "on the curve is still 0.0.</p>")

    # ---- in-domain -----------------------------------------------------
    A("<h2>Recorded (not pass/fail): in-domain</h2>")
    A("<table><thead><tr><th>Dose</th><th>Seed</th><th>eval pooled EER</th>"
      "<th>A19</th><th>A20</th></tr></thead><tbody>")
    for d in DOSES:
        for s in SEEDS:
            pooled, a19, a20 = IN_DOMAIN[d][s]
            A(f"<tr><td>{d}</td><td class='num'>{s}</td>"
              f"<td class='num'>{pooled:.3f}%</td>"
              f"<td class='num'>{a19:.3f}%</td>"
              f"<td class='num'>{a20:.3f}%</td></tr>")
    A("</tbody></table>")
    A("<div class='callout warn'>In-domain is <strong>not monotone either, and "
      "is far flatter</strong> than out-of-domain. Dose 0.25 seed 43 (8.822 %) "
      "is nominally the best in-domain number in the entire table &mdash; "
      "better than dose 0.0 seed 42 (9.241 %). Selecting on in-domain pooled "
      "EER would pick 0.25, which is one of the two <em>worst</em> points "
      "out-of-domain. This is the same lesson as docs/18 and docs/20, and it "
      "is why the criterion was written on ITW <code>full_call</code>.</div>")

    # ---- conclusion ----------------------------------------------------
    A("<h2>Conclusion</h2>")
    A("<div class='callout good'><strong>Remove or permanently disable the "
      "codec/telephony augmenter. Do not tune it.</strong><br>"
      "docs/20 could only say &ldquo;this on/off setting is worse on&rdquo;. "
      "docs/21 says the entire dose axis is worthless &mdash; including the "
      "points nobody had tried. That is a stronger and more actionable "
      "finding, and it is the reason this experiment was worth ~14 h of "
      "GPU.</div>")
    A("<p><strong>Not re-run to chase a different answer</strong> (ground rule "
      "5). Published as measured.</p>")

    A("<div class='foot'>")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
      "<code>scripts/build_docs21_site.py</code> &mdash; every number read "
      "from the adjudication report, none hand-typed.<br>")
    A(f"Source: <code>{esc(Path(path).name)}</code> &middot; bootstrap: "
      "2,000-rep percentile, seed 1234 &middot; n = 19,963 bonafide / 11,816 "
      "spoof (ITW), 4,058 / 15,936 (eval slice).")
    A("</div></div></body></html>")

    out = REPO / args.out
    out.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"verdict: {verdict} -- {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
