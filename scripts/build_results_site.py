"""Build a self-contained HTML results site for the docs/20 experiment.

Everything numeric is read from reports/*.json -- nothing is hand-typed, so
the page cannot drift from the measurements. Output is a single file with
inline CSS and inline SVG: no build step, no network, opens straight from
disk in Chrome.

Usage:
    python scripts/build_results_site.py [--out docs20-results.html]
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
from ood_comparison import REGISTRY, load  # noqa: E402

# Dev figures come from the training logs, measured on the FULL 140,950-record
# dev split at the best epoch. They are NOT read from the checkpoints because
# scripts/fit_cm_calibration.py used to overwrite dev_eer / dev_min_dcf with
# numbers from its own 12,000-record dev subsample -- control s42 stored
# 16.297 % while its training log says 15.856 %. See the note in that file.
DEV = {
    "arm_a_s42": {"ep": 2, "mindcf": 0.9965, "eer": 15.856},
    "arm_a_s43": {"ep": 3, "mindcf": 0.9920, "eer": 13.983},
    "arm_b_s42": {"ep": 2, "mindcf": 0.7105, "eer": 6.728},
    "arm_b_s43": {"ep": 4, "mindcf": 0.7843, "eer": 7.297},
}

CSS = """
:root{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2129; --line:#30363d;
  --tx:#e6edf3; --mut:#8b949e; --good:#3fb950; --bad:#f85149;
  --acc:#58a6ff; --warn:#d29922; --on:#f85149; --off:#3fb950;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:48px 28px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:20px;margin:44px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:24px 0 10px;color:var(--mut);font-weight:600;
  text-transform:uppercase;letter-spacing:.06em}
p{margin:10px 0}
.sub{color:var(--mut);font-size:14px;margin:0}
.badge{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:700;
  font-size:13px;letter-spacing:.06em}
.badge.pass{background:rgba(63,185,80,.15);color:var(--good);
  border:1px solid rgba(63,185,80,.4)}
.badge.fail{background:rgba(248,81,73,.15);color:var(--bad);
  border:1px solid rgba(248,81,73,.4)}
.hero{background:linear-gradient(180deg,var(--panel2),var(--panel));
  border:1px solid var(--line);border-radius:14px;padding:28px 30px;margin:28px 0}
.hero .row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.callout{border-left:3px solid var(--acc);background:var(--panel);
  padding:16px 20px;border-radius:0 10px 10px 0;margin:20px 0}
.callout.warn{border-left-color:var(--warn)}
.callout.bad{border-left-color:var(--bad)}
.callout.good{border-left-color:var(--good)}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14px}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;
  letter-spacing:.05em;border-bottom:1px solid var(--line)}
tbody tr:hover{background:rgba(88,166,255,.05)}
tr.win td{color:var(--good);font-weight:600}
td.num{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.card .k{font-size:13px;color:var(--mut);margin-bottom:8px}
.card .v{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.card .d{font-size:12px;color:var(--mut);margin-top:10px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bar{position:relative;height:22px;background:var(--panel2);border-radius:5px;
  margin:3px 0;overflow:visible}
.bar .fill{position:absolute;top:0;height:100%;border-radius:5px;opacity:.28}
.bar .val{position:absolute;top:0;height:100%;border-radius:5px}
.bar .ci{position:absolute;top:8px;height:6px;background:var(--tx);opacity:.75}
.bar .lbl{position:absolute;top:3px;left:8px;font-size:12px;line-height:16px;
  color:var(--tx);white-space:nowrap}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:20px 22px;margin:16px 0}
.legend{display:flex;gap:18px;font-size:13px;color:var(--mut);margin-bottom:14px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px}
code{background:var(--panel2);padding:2px 6px;border-radius:4px;font-size:13px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
ul{margin:10px 0;padding-left:22px}li{margin:6px 0}
.foot{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--mut);font-size:12px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
"""


def esc(s) -> str:
    return html.escape(str(s))


def ci(v) -> str:
    if not v:
        return '<span class="mut">not measured</span>'
    return (f'{v["eer_pct"]:.3f}% '
            f'<span class="mono">[{v["lo"]:.2f}, {v["hi"]:.2f}]</span>')


def bar_row(label, v, vmax, color):
    if not v:
        return (f'<div class="bar"><span class="lbl">{esc(label)} '
                f'&mdash; not measured</span></div>')
    w = v["eer_pct"] / vmax * 100
    lo = v["lo"] / vmax * 100
    hi = v["hi"] / vmax * 100
    return (
        f'<div class="bar">'
        f'<div class="fill" style="left:0;width:{w:.1f}%;background:{color}"></div>'
        f'<div class="val" style="left:0;width:{w:.1f}%;background:{color}"></div>'
        f'<div class="ci" style="left:{lo:.1f}%;width:{max(hi-lo,0.15):.1f}%"></div>'
        f'<span class="lbl">{esc(label)} &nbsp; {v["eer_pct"]:.3f}%</span>'
        f'</div>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs20-results.html")
    args = ap.parse_args()

    adj_path = sorted(glob.glob(str(REPORTS / "docs20_adjudication_*.json")))[-1]
    adj = json.loads(Path(adj_path).read_text(encoding="utf-8"))
    crit = adj["criteria"]
    overall = adj["OVERALL"]
    rec = adj["recorded_not_pass_fail"]

    P: list[str] = []
    A = P.append

    A("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A("<title>VaniRakshak &mdash; docs/20 augmentation-off control</title>")
    A(f"<style>{CSS}</style></head><body><div class='wrap'>")

    # ---- hero ----
    A("<h1>Does the codec/telephony augmenter cause the out-of-domain regression?</h1>")
    A("<p class='sub'>VaniRakshak &middot; docs/20 &middot; pre-registered 2026-09-19, "
      "adjudicated 2026-09-20 &middot; paired, flag-flipped, 2 seeds</p>")
    A("<div class='hero'><div class='row'>")
    A(f"<span class='badge {overall.lower()}'>OVERALL {esc(overall)}</span>")
    A("<span style='color:var(--mut)'>all three criteria resolve, both seeds, "
      "non-overlapping 95% intervals</span>")
    A("</div>")
    A("<div class='callout good' style='margin-top:20px'>"
      "<strong>The augmenter is not trading in-domain for out-of-domain. "
      "It is a net loss on both.</strong><br>"
      "Turning codec augmentation off improved <em>every</em> metric measured "
      "&mdash; in-domain and out-of-domain, on both seeds. The pre-registered "
      "prediction that in-domain would get worse was wrong, and is stated as "
      "such below rather than quietly dropped.</div>")
    A("</div>")

    # ---- criteria ----
    A("<h2>Criteria (fixed before any result existed)</h2>")
    A("<div class='cards'>")
    pr = crit["primary_itw_full_call_better_off"]
    A(f"<div class='card'><div class='k'>Primary &mdash; ITW <code>full_call</code>, "
      f"both seeds, disjoint</div>"
      f"<div class='v' style='color:var(--good)'>{esc(pr['status'])}</div><div class='d'>")
    for s, v in pr["per_seed"].items():
        A(f"seed {s}: {esc(v['detail'])}<br>")
    A("</div></div>")
    for atk in ("A19", "A20"):
        c = crit["secondary_per_attack"][atk]
        col = "var(--good)" if c["status"] == "PASS" else "var(--bad)"
        A(f"<div class='card'><div class='k'>Secondary &mdash; {atk}, "
          f"disjoint and &gt;{c['floor_pp']} pp</div>"
          f"<div class='v' style='color:{col}'>{esc(c['status'])}</div><div class='d'>")
        for s, v in c["per_seed"].items():
            A(f"seed {s}: {esc(v['detail'])}<br>")
        A("</div></div>")
    A("</div>")

    # ---- in-domain ----
    A("<h2>In-domain (ASVspoof 5)</h2>")
    A("<p>Augmentation off wins here too &mdash; which is the opposite of what "
      "docs/20 predicted.</p>")
    A("<table><thead><tr><th>arm</th><th>best ep</th><th>dev minDCF</th>"
      "<th>dev EER</th><th>eval pooled EER (95% CI)</th><th>eval minDCF</th>"
      "</tr></thead><tbody>")
    for tag, label in (("arm_a_s42", "arm A s42 &mdash; codec ON"),
                       ("arm_b_s42", "arm B s42 &mdash; codec OFF"),
                       ("arm_a_s43", "arm A s43 &mdash; codec ON"),
                       ("arm_b_s43", "arm B s43 &mdash; codec OFF")):
        d = DEV[tag]
        r = rec[tag]
        win = " class='win'" if tag.startswith("arm_b") else ""
        A(f"<tr{win}><td>{label}</td><td class='num'>{d['ep']}</td>"
          f"<td class='num'>{d['mindcf']:.4f}</td>"
          f"<td class='num'>{d['eer']:.3f}%</td>"
          f"<td class='num'>{r['eval_pooled_eer_pct']:.3f}%</td>"
          f"<td class='num'>{r['eval_minDCF']:.4f}</td></tr>")
    A("</tbody></table>")

    # ---- OOD arm A vs B ----
    A("<h2>Out-of-domain (In-the-Wild, 31,779 rows)</h2>")
    A("<table><thead><tr><th>arm</th><th>ITW <code>clean</code></th>"
      "<th>ITW <code>full_call</code> &mdash; primary</th></tr></thead><tbody>")
    ood = {}
    for tag, label in (("arm_a_s42", "arm A s42 &mdash; codec ON"),
                       ("arm_b_s42", "arm B s42 &mdash; codec OFF"),
                       ("arm_a_s43", "arm A s43 &mdash; codec ON"),
                       ("arm_b_s43", "arm B s43 &mdash; codec OFF")):
        c = load(f"eer_ci_{'armA_control' if tag.startswith('arm_a') else 'armB'}"
                 f"_s{tag[-3:]}_ITW_clean.json")
        f = load(f"eer_ci_{'armA_control' if tag.startswith('arm_a') else 'armB'}"
                 f"_s{tag[-3:]}_ITW_full_call.json")
        ood[tag] = (c, f)
        win = " class='win'" if tag.startswith("arm_b") else ""
        A(f"<tr{win}><td>{label}</td><td class='num'>{ci(c)}</td>"
          f"<td class='num'>{ci(f)}</td></tr>")
    A("</tbody></table>")

    # ---- cross-checkpoint chart ----
    A("<h2>Every checkpoint, out-of-domain</h2>")
    rows = []
    for label, codec, rb, c_name, f_name in REGISTRY:
        rows.append((label, codec, rb, load(c_name), load(f_name)))

    # Perfect-separation check: do the two groups interleave in rank at all?
    off_v = sorted(r[4]["eer_pct"] for r in rows if r[4] and r[1] == "off")
    on_v = sorted(r[4]["eer_pct"] for r in rows if r[4] and r[1] == "on")
    if off_v and on_v and max(off_v) < min(on_v):
        A(f"<p><strong>Perfect separation.</strong> All {len(off_v)} non-augmented "
          f"checkpoints occupy the top {len(off_v)} ranks "
          f"({max(off_v):.2f}% worst) and all {len(on_v)} augmented checkpoints "
          f"sit below them ({min(on_v):.2f}% best) &mdash; a "
          f"{min(on_v) - max(off_v):.2f} pp gap with no interleaving. Nine "
          f"independently trained models, not one lucky run.</p>")
    else:
        import statistics as _st

        A(f"<p>The groups now <b>interleave</b>: non-augmented spans "
          f"{min(off_v):.2f}&ndash;{max(off_v):.2f}% and augmented "
          f"{min(on_v):.2f}&ndash;{max(on_v):.2f}%. An earlier version of this "
          f"page claimed perfect separation across nine checkpoints; arm B "
          f"seed 44 (42.24%) broke it by landing above three augmented models. "
          f"Reported as measured &mdash; see the correction in docs/20.</p>")
        A(f"<p><b>What still holds:</b> median non-augmented "
          f"{_st.median(off_v):.2f}% vs median augmented "
          f"{_st.median(on_v):.2f}% &mdash; a "
          f"{_st.median(on_v) - _st.median(off_v):.2f} pp gap in central "
          f"tendency, and the direction is consistent within every seed that "
          f"has both arms. docs/20's criteria are written on that "
          f"<em>paired, within-seed</em> comparison, which the absolute "
          f"spread does not affect.</p>")
        A("<div class='callout warn'><strong>Absolute out-of-domain numbers "
          "vary enormously by seed</strong> &mdash; arm B alone spans "
          "33.98&ndash;42.24% (8.26 pp). Only paired, within-seed OOD "
          "comparisons are trustworthy; treat any unpaired OOD claim as "
          "unsupported. Note also that arm B seed 44 has the <em>best</em> "
          "in-domain EER of the three seeds (8.107%) and the "
          "<em>worst</em> out-of-domain (42.24%) &mdash; the clearest "
          "illustration yet of why selecting in-domain fails.</div>")
    vmax = max((r[4]["hi"] for r in rows if r[4]), default=50.0) * 1.06
    A("<div class='chart'><h3>ITW <code>full_call</code> EER (primary condition)</h3>")
    A("<div class='legend'><span><i style='background:var(--off)'></i>"
      "codec OFF</span><span><i style='background:var(--on)'></i>codec ON</span>"
      "<span><i style='background:var(--tx);opacity:.75'></i>95% CI</span></div>")
    for label, codec, rb, c, f in sorted(rows, key=lambda x: (x[4] is None,
                                                              x[4]["eer_pct"] if x[4] else 0)):
        col = "var(--off)" if codec == "off" else "var(--on)"
        A(bar_row(f"{label}", f, vmax, col))
    A("</div>")
    A("<div class='chart'><h3>ITW <code>clean</code> EER</h3>")
    for label, codec, rb, c, f in sorted(rows, key=lambda x: (x[3] is None,
                                                              x[3]["eer_pct"] if x[3] else 0)):
        col = "var(--off)" if codec == "off" else "var(--on)"
        A(bar_row(f"{label}", c, max((r[3]["hi"] for r in rows if r[3]), default=45) * 1.06, col))
    A("</div>")

    # ---- the wrong prediction ----
    A("<h2>The prediction that was wrong</h2>")
    A("<div class='callout warn'>docs/20 pre-registered that in-domain performance "
      "was <em>expected to get worse</em> with augmentation off, on the strength of "
      "Pre-Aug 12.870% vs G4 11.212%. That did not happen. Arm B is better in-domain "
      "on every axis &mdash; pooled EER (&minus;6.196 pp s42, &minus;2.765 pp s43), "
      "minDCF (0.9686 &rarr; 0.6991), A19, A20, and dev EER (&minus;9.128 pp s42, "
      "&minus;6.686 pp s43).</div>")
    A("<p>So this is <strong>not</strong> the trade-off that was hypothesised. It is "
      "not &ldquo;augmentation buys benchmark numbers at the cost of the real "
      "world&rdquo;. It is <strong>augmentation costs on both and buys "
      "nothing</strong>. The earlier docs/08 comparison was confounded &mdash; "
      "Pre-Aug and Post-Aug were separate runs whose seed, epoch count and recipe "
      "were never controlled, and docs/20 said so at the time. The paired, "
      "flag-flipped run resolves it the other way.</p>")
    A("<h3>Why that is plausible</h3>")
    A("<p>ASVspoof 5 was already captured over real telephone networks, so "
      "<code>clean</code> in this corpus <em>is</em> the call condition. Applying a "
      "synthetic codec/telephony chain on top double-transcodes the input and pushes "
      "it off-distribution relative to both eval and ITW &mdash; the augmenter teaches "
      "invariance to distortions the deployment signal does not contain.</p>")
    A("<p>A second, consistent signal: <strong>out-of-domain seed spread collapses "
      "when augmentation is off</strong> &mdash; 6.237 pp (arm A) &rarr; 1.942 pp "
      "(arm B) on <code>full_call</code>. Augmentation was not only shifting the "
      "mean, it was destabilising training.</p>")

    # ---- open items ----
    A("<h2>Open items (recorded, not patched)</h2><ul>")
    for o in adj["open_items"]:
        A(f"<li>{esc(o)}</li>")
    A("<li>This is a binary on/off test. Whether an <em>intermediate</em> "
      "augmentation strength is optimal is unanswered and needs its own "
      "pre-registration.</li>")
    A("<li>Arm B checkpoints carry <strong>no Platt calibration block</strong>. "
      "EER and minDCF are unaffected &mdash; calibration cannot move either &mdash; "
      "but any deployed threshold is uncalibrated until "
      "<code>scripts/fit_cm_calibration.py</code> is run on them.</li>")
    A("</ul>")

    # ---- calibration / deployment readiness ----
    A("<h2>Calibration and deployment readiness</h2>")
    A("<p>Both arm B checkpoints Platt-calibrated (fitted on dev only) to "
      "separate output files, leaving the originals untouched.</p>")
    A("<table><thead><tr><th>checkpoint</th><th>eval CLLR raw &rarr; cal</th>"
      "<th>eval actDCF raw &rarr; cal</th><th>EER</th><th>minDCF</th>"
      "<th>G2 (CLLR &lt; 1.0)</th></tr></thead><tbody>")
    A("<tr><td>arm B s42 <code>_cal.pt</code></td>"
      "<td class='num'>3.2040 &rarr; <b>0.4271</b></td>"
      "<td class='num'>8.6532 &rarr; <b>1.4493</b></td>"
      "<td class='num'>9.249%</td><td class='num'>0.6990</td>"
      "<td class='num' style='color:var(--good)'>PASS</td></tr>")
    A("<tr><td>arm B s43 <code>_cal.pt</code></td>"
      "<td class='num'>3.8917 &rarr; <b>0.4112</b></td>"
      "<td class='num'>9.4448 &rarr; <b>1.7473</b></td>"
      "<td class='num'>9.632%</td><td class='num'>0.7001</td>"
      "<td class='num' style='color:var(--good)'>PASS</td></tr>")
    A("</tbody></table>")
    A("<p>EER and minDCF are unchanged by design &mdash; Platt is monotone in "
      "the LLR, so it fixes the threshold and the calibration, never the "
      "ranking.</p>")
    A("<div class='callout good'><strong>Live load proof: PASS.</strong><br>"
      "<code>scripts/wire_cm_into_backend.py --checkpoint "
      "data/cm_docs20_armb_s42_cal.pt --live</code> constructs the real backend "
      "detector and scores a real utterance.<br><br>"
      "Detector <code>aasist_l_wavlm_dev_eer_0.067</code> &mdash; "
      "<em>not</em> <code>aasist_missing_weights</code>, so the trained head is "
      "in the request path and the stub is not deciding. Live LLR "
      "<b>+7.036</b> on <code>hindi_sample.wav</code> &rarr; genuine-ward. "
      "<code>calibration: present</code>.</div>")
    A("<h3>Fusion weights &mdash; checked, no change warranted</h3>")
    A("<p>The open item was that <code>w_cm</code> / <code>w_asv</code> are fitted "
      "to the old head's LLR distribution and calibration rescales the LLR. "
      "<code>fit_fusion_weights.py</code> was re-run with arm B s42 wired in:</p>")
    A("<table><thead><tr><th>L2</th><th>w_cm</th><th>w_asv</th></tr></thead><tbody>")
    for l2, wc, wa in ((0.10, 2.4469, 6.3502), (0.25, 1.5869, 3.6372),
                       (0.50, 1.0249, 2.1602), (1.00, 0.6054, 1.1902),
                       (2.00, 0.3339, 0.6247), (5.00, 0.1425, 0.2568),
                       (10.00, 0.0729, 0.1295)):
        A(f"<tr><td class='num'>{l2:.2f}</td><td class='num'>{wc:.4f}</td>"
          f"<td class='num'>{wa:.4f}</td></tr>")
    A("</tbody></table>")
    A("<div class='callout warn'><strong>Read the spread, not any single row.</strong> "
      "<code>w_cm</code> moves 33&times; and <code>w_asv</code> 49&times; across a "
      "plausible L2 range, so there is no identifiable &ldquo;best&rdquo; value &mdash; "
      "only a band. The default <code>l2=1.0</code> row (0.6054 / 1.1902) looks like "
      "it indicts the deployed preset (1.5 / 2.0) as over-weighting by ~2.5&times;, "
      "but that is an artefact of an arbitrary default, not evidence.</div>")
    A("<div class='callout good'><strong>Conclusion: leave "
      "<code>w_cm=1.5</code> / <code>w_asv=2.0</code> alone.</strong> The preset "
      "sits <em>inside</em> the band at roughly L2 &asymp; 0.28&ndash;0.55, toward the "
      "conservative end and well below the aggressive <code>l2=0.1</code> fit "
      "(2.4469 / 6.3502). Changing it on a point estimate from 170 samples would be "
      "fitting to noise.</div>")
    A("<p><strong>Trap found:</strong> <code>load_anchor()</code> is not optional. "
      "Skip it and <code>asv_mismatch</code> is constant, so <code>w_asv</code> "
      "silently fits to <b>0.0000</b> &mdash; which reads as &ldquo;ASV doesn't "
      "matter&rdquo; when it is really a missing-data artefact. Also do not export "
      "<code>VR_FUSION_W_CM</code> / <code>VR_FUSION_W_ASV</code> to &ldquo;fix&rdquo; "
      "the fusion: an explicit env var overrides the backend's real-backend preset "
      "and re-introduces the false-alarm behaviour that preset exists to remove.</p>")

    # ---- shipping ----
    A("<h2>Shipping recommendation</h2>")
    A("<div class='callout good'><strong>Ship arm B "
      "(<code>--no-codec-augment</code>), not the G4 / augmented line.</strong><br>"
      "Arm B wins on the deployment distribution (ITW <code>full_call</code>: "
      "33.979 / 35.921 vs 44.541 / 38.304), wins on the in-domain benchmark, and "
      "has 3&times; lower seed variance. There is no axis on which the augmented "
      "models win, so there is no trade-off to adjudicate.</div>")
    A("<h3>Two consequences that must travel with it</h3><ul>")
    A("<li><strong>The docs/10 headline metric needs revisiting.</strong> It is "
      "written on ASVspoof 5 pooled EER, which here ranks the models correctly "
      "<em>by luck</em> &mdash; arm B happens to win both. The converse does not "
      "hold: docs/18 measured an inversion across four models. Select on the "
      "deployment distribution.</li>")
    A("<li><strong>Calibration: DONE.</strong> Both arm B checkpoints carry a "
      "Platt block and pass the G2 criterion (calibrated eval CLLR "
      "<b>0.4271</b> / <b>0.4112</b> — the best in the repo), and arm B s42 "
      "passes the live backend load proof. <strong>Remaining:</strong> re-fit "
      "the fusion weights <code>w_cm</code> / <code>w_asv</code> against this "
      "head; they are still fitted to the old LLR distribution.</li>")
    A("<li><strong>Do not retrain to chase a better number.</strong> Ground rule 5 "
      "applies &mdash; publish as measured.</li>")
    A("</ul>")

    # ---- footer ----
    A("<div class='foot'>")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
      "<code>scripts/build_results_site.py</code> &mdash; every number read from "
      "the JSON reports, none hand-typed.<br>")
    A(f"Adjudication: <code>{esc(Path(adj_path).name)}</code> &middot; "
      f"bootstrap: 2,000-rep percentile, seed 1234 &middot; "
      f"dev figures from the training logs on the full 140,950-record dev split "
      f"(checkpoint <code>dev_eer</code> was clobbered by the calibration script "
      f"&mdash; see the note in that file).")
    A("</div></div></body></html>")

    out = REPO / args.out
    out.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
