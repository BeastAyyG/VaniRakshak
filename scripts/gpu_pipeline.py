"""Resumable GPU-box pipeline for the docs/15 "Next" items.

Stages, in order (each is idempotent; a finished stage is skipped on re-run):

1. ``t1_calibrate``   docs/14 §6 step 1: choose every threshold on the DEV dump,
                      measure on the GATE-split dump (both already scored by
                      Phase 1/2; no new scoring needed).
2. ``t1_grade``       grade P1-S and P1-F exactly as docs/13 Phase 1 wrote them,
                      and report genuine false-BLOCK (not pre-registered).
3. ``t2_cuda``        docs/14 §6 step 2 / P1-T: integration_check.py with ECAPA,
4. ``t2_cpu``         the AASIST CM at the fitted operating point and the fitted
                      thresholds, on CUDA fp16 and on CPU fp32.
5. ``d2_telephony``   docs/13 §2.3 / STOP-C: the incumbent scored on a fixed,
                      seeded gate-split subset, clean and through G.711u,
                      AMR-NB 12.2k and AMR-WB 12.65k. Transcoding happens in
                      memory (C: is under the 50 GB floor docs/13 sets for
                      writing audio); only scores are kept, in shards of 1024
                      rows, so an interrupted run resumes where it stopped.

State lives in ``--root`` (default C:/vr-out/pipeline/state.json) and is what
``scripts/gpu_dashboard.py`` shows. No stage trains a model: docs/13 STOP-B
and docs/15 forbid CM training until a docs/16 pre-registration exists.

Usage:
    python scripts/gpu_pipeline.py                 # run / resume every stage
    python scripts/gpu_pipeline.py --only d2_telephony
"""

from __future__ import annotations

import argparse
import audioop
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = Path(os.environ.get("VR_PIPELINE_OUT", "C:/vr-out"))
DEV_DUMP = OUT / "p1" / "dev" / "incumbent.npz"
GATE_DUMP = OUT / "p2" / "gate" / "incumbent.npz"
GATE_MANIFEST = Path("A:/cm-corpus/manifests/gate_split_608k.tsv")
INCUMBENT = REPO / "data" / "cm_aasist.pt"
T1_DIR = OUT / "t1"
D2_DIR = OUT / "d2"
SUBSET_SEED = 20260928
PER_ATTACK = 1000
BONAFIDE = 4000
SHARD = 1024
CONDITIONS = ["clean", "g711u", "amrnb_12k2", "amrwb_12k65"]


# --------------------------------------------------------------------------- state


class State:
    def __init__(self, root: Path, filename: str = "state.json") -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / filename
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"stages": {}}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def stage(self, name: str) -> dict:
        return self.data["stages"].setdefault(name, {"status": "pending"})

    def update(self, name: str, **kw) -> None:
        self.stage(name).update(kw)
        self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()


class Ctx:
    def __init__(self, state: State, name: str) -> None:
        self.state, self.name = state, name
        self.log_path = state.root / f"{name}.log"

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(f"{self.name}: {line}", flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def progress(self, done: int, total: int, message: str = "") -> None:
        self.state.update(self.name, progress={"done": done, "total": total}, message=message)

    def run(self, cmd: List[str], env: Optional[dict] = None, timeout: int = 7200) -> subprocess.CompletedProcess:
        self.log("$ " + " ".join(cmd))
        full_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                    **(env or {})}
        with open(self.log_path, "a", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=REPO, env=full_env, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, text=True)
        self.log(f"exit {p.returncode}")
        return p


# --------------------------------------------------------------------------- T1


def t1_calibrate(ctx: Ctx) -> dict:
    out = T1_DIR / "decision_layer_calibration.json"
    op_ckpt = T1_DIR / "cm_aasist_op.pt"
    if out.exists() and op_ckpt.exists():
        ctx.log("already done")
        return {"calibration": str(out), "op_checkpoint": str(op_ckpt)}
    T1_DIR.mkdir(parents=True, exist_ok=True)
    for p in (out, op_ckpt):  # a half-finished earlier attempt
        if p.exists():
            p.unlink()
    ctx.progress(0, 1, "simulating sessions on dev, measuring on gate")
    p = ctx.run([PY, "scripts/calibrate_decision_layer.py",
                 "--dev-scores", str(DEV_DUMP), "--eval-scores", str(GATE_DUMP),
                 "--challenge-consecutive", "3", "--out", str(out),
                 "--checkpoint", str(INCUMBENT), "--write-checkpoint", str(op_ckpt)])
    if p.returncode != 0:
        raise RuntimeError("calibrate_decision_layer.py failed; see log")
    ctx.progress(1, 1, "done")
    return {"calibration": str(out), "op_checkpoint": str(op_ckpt)}


def _worst_attack(cell: dict) -> float:
    return float(cell.get("spoof_p_block_worst_attack", min(cell["spoof_p_block_by_end"].values())))


def t1_grade(ctx: Ctx) -> dict:
    cal = json.loads((T1_DIR / "decision_layer_calibration.json").read_text())
    ev = cal["results"]["eval"]
    cells = list(ev["after"].keys())
    rows = {}
    for c in cells:
        b, a = ev["before"][c], ev["after"][c]
        rows[c] = {
            "spoof_block_worst_attack_before": _worst_attack(b),
            "spoof_block_worst_attack_after": _worst_attack(a),
            "p1s_rise_pp": round(100 * (_worst_attack(a) - _worst_attack(b)), 2),
            "genuine_challenge_60s_before": b["genuine"]["p_challenge_within_60s"],
            "genuine_challenge_60s_after": a["genuine"]["p_challenge_within_60s"],
            "p1f_rise_pp": round(100 * (a["genuine"]["p_challenge_within_60s"]
                                        - b["genuine"]["p_challenge_within_60s"]), 2),
            "genuine_block_by_end_after": a["genuine"]["p_block_by_end"],
        }
    # "Worst cell" = the grid cell where BLOCK|spoof is lowest after the change.
    worst = min(cells, key=lambda c: rows[c]["spoof_block_worst_attack_after"])
    p1s_min_rise = min(r["p1s_rise_pp"] for r in rows.values())
    grade = {
        "criteria": "docs/13 Phase 1: P1-S rise >= 30 pp at the worst ASV x channel cell; "
                    "P1-F rise of P(challenge within 60 s | genuine) <= 5 pp",
        "eval_source": str(GATE_DUMP),
        "chosen": cal["chosen"],
        "worst_cell": worst,
        "p1s_rise_at_worst_cell_pp": rows[worst]["p1s_rise_pp"],
        "p1s_min_rise_any_cell_pp": p1s_min_rise,
        "P1_S_pass": rows[worst]["p1s_rise_pp"] >= 30.0,
        "P1_F_pass": all(r["p1f_rise_pp"] <= 5.0 for r in rows.values()),
        "p1f_max_rise_pp": max(r["p1f_rise_pp"] for r in rows.values()),
        "not_pre_registered": {
            "genuine_block_by_end_after_max": max(r["genuine_block_by_end_after"] for r in rows.values()),
            "note": "Genuine false-BLOCK per call is not a docs/13 criterion. The dev budget "
                    "(--max-false-block) is 1 % at ASV 1.0; eval values above it are dev->eval shift.",
        },
        "cells": rows,
        "assumption": cal.get("assumption"),
    }
    (T1_DIR / "t1_grade.json").write_text(json.dumps(grade, indent=2), encoding="utf-8")
    ctx.log(f"P1-S {'PASS' if grade['P1_S_pass'] else 'FAIL'} ({grade['p1s_rise_at_worst_cell_pp']} pp at "
            f"{worst}); P1-F {'PASS' if grade['P1_F_pass'] else 'FAIL'} (max rise {grade['p1f_max_rise_pp']} pp); "
            f"genuine BLOCK after, max {grade['not_pre_registered']['genuine_block_by_end_after_max']}")
    ctx.progress(1, 1, "graded")
    return {"grade": str(T1_DIR / "t1_grade.json"), "P1_S_pass": grade["P1_S_pass"],
            "P1_F_pass": grade["P1_F_pass"]}


# --------------------------------------------------------------------------- T2


def _t2_env() -> Dict[str, str]:
    chosen = json.loads((T1_DIR / "decision_layer_calibration.json").read_text())["chosen"]
    env = {
        "VR_ASV_BACKEND": "ecapa",
        "VR_CM_BACKEND": "aasist",
        "VR_CM_AASIST_WEIGHTS": str(T1_DIR / "cm_aasist_op.pt"),
        "VR_AUDIT_PATH": str(OUT / "pipeline" / "t2_audit.jsonl"),
    }
    for k in ("VR_CLONE_CM_LLR_MAX", "VR_CLONE_MIN_WINDOWS", "VR_CHALLENGE_CM_LLR_MAX",
              "VR_CHALLENGE_MIN_WINDOWS", "VR_CHALLENGE_TRIGGER_CONSECUTIVE"):
        if chosen.get(k) is not None:
            env[k] = str(chosen[k])
    return env


def _t2(ctx: Ctx, device: str) -> dict:
    env = _t2_env()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    ctx.progress(0, 1, f"probing loaded backends on {device}")
    probe = ("import sys; sys.path.insert(0, 'vanirakshak-backend'); "
             "from vanirakshak.detectors.pipeline import DetectionPipeline as P; p = P(); "
             "print('LOADED_BACKENDS', p.asv.name, p.cm.name, getattr(p.cm, 'operating_threshold', None))")
    ctx.run([PY, "-c", probe], env=env, timeout=1800)
    ctx.progress(0, 1, f"integration_check on {device}")
    ctx.run([PY, "scripts/integration_check.py"], env=env, timeout=3600)
    text = ctx.log_path.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"(\d+)/(\d+) checks passed", text)
    passed, total = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)

    def ok(pat: str) -> Optional[bool]:
        hits = re.findall(r"\[(PASS|FAIL)\] " + pat, text)
        return None if not hits else hits[-1] == "PASS"

    p1t = {
        "demo_01_allow": ok(r"demo_01_genuine\.wav tier == ALLOW"),
        "demo_02_not_block": ok(r"demo_02_degraded_copy\.wav tier"),
        "demo_04_block_every_window": ok(r"demo_04_impostor BLOCKs on every window"),
    }
    # demo_02's tier check expects ALLOW; P1-T only needs "not BLOCK".
    tier02 = re.findall(r"demo_02_degraded_copy\.wav tier == \w+ .*?got tier=(\w+)", text)
    if tier02:
        p1t["demo_02_not_block"] = tier02[-1] != "BLOCK"
    # The CM silently degrades to the stub if the weights fail to load; a P1-T
    # on the stub would prove nothing, so require the real backend by name.
    loaded = re.findall(r"LOADED_BACKENDS (\S+) (\S+) (\S+)", text)
    asv_b, cm_b, op = loaded[-1] if loaded else ("?", "?", "?")
    real = "ecapa" in asv_b and cm_b.startswith("aasist") and "missing" not in cm_b
    res = {"device": device, "checks_passed": passed, "checks_total": total, "p1t": p1t,
           "backends": {"asv": asv_b, "cm": cm_b, "cm_operating_threshold": op},
           "P1_T_pass": real and all(v is True for v in p1t.values()), "env": env}
    ctx.log(json.dumps(res))
    ctx.progress(1, 1, f"{passed}/{total} checks, P1-T {'PASS' if res['P1_T_pass'] else 'FAIL'}")
    return res


def t2_cuda(ctx: Ctx) -> dict:
    return _t2(ctx, "cuda")


def t2_cpu(ctx: Ctx) -> dict:
    return _t2(ctx, "cpu")


# --------------------------------------------------------------------------- D2 / STOP-C


def _ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for p in base.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"):
        return str(p)
    return None


def _ffmpeg_roundtrip(ffmpeg: str, x16: np.ndarray, codec: str) -> np.ndarray:
    """16 kHz int16 -> codec -> 16 kHz int16, all through pipes."""
    if codec == "amrnb_12k2":
        enc = ["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-f", "amr"]
    else:
        enc = ["-ar", "16000", "-c:a", "libvo_amrwbenc", "-b:a", "12.65k", "-f", "amr"]
    base = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    coded = subprocess.run(base + ["-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0"] + enc + ["pipe:1"],
                           input=x16.tobytes(), capture_output=True, check=True).stdout
    pcm = subprocess.run(base + ["-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
                         input=coded, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).copy()


def _g711u(x16: np.ndarray) -> np.ndarray:
    """Bit-exact G.711 mu-law via audioop, at 8 kHz, back to 16 kHz."""
    from scipy.signal import resample_poly

    x8 = np.clip(resample_poly(x16.astype(np.float32), 1, 2), -32768, 32767).astype(np.int16)
    dec = np.frombuffer(audioop.ulaw2lin(audioop.lin2ulaw(x8.tobytes(), 2), 2), dtype=np.int16)
    y = resample_poly(dec.astype(np.float32), 2, 1)
    return np.clip(y, -32768, 32767).astype(np.int16)


def _subset() -> dict:
    d = np.load(GATE_DUMP)
    attack, label = d["attack"].astype(str), d["label"].astype(int)
    rng = np.random.default_rng(SUBSET_SEED)
    picks = []
    bon = np.flatnonzero(label == 1)
    picks.append(np.sort(rng.choice(bon, min(BONAFIDE, bon.size), replace=False)))
    for a in sorted(set(attack[label == 0])):
        idx = np.flatnonzero((label == 0) & (attack == a))
        picks.append(np.sort(rng.choice(idx, min(PER_ATTACK, idx.size), replace=False)))
    rows = np.sort(np.concatenate(picks))
    return {"dump_pos": rows, "row_index": d["row_index"][rows], "label": label[rows],
            "attack": attack[rows], "gate_raw_llr": d["raw_llr"][rows],
            "manifest_sha256": str(d["manifest_sha256"])}


def d2_telephony(ctx: Ctx) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import soundfile as sf
    import torch

    from cm_frontend import assert_frozen_checkpoint, load_frontend_for
    from ml.aasist_model import AASISTLHead
    from ml.augment import prepare_audio_window

    ffmpeg = _ffmpeg()
    conditions = [c for c in CONDITIONS if not c.startswith("amr") or ffmpeg]
    if len(conditions) < len(CONDITIONS):
        ctx.log("ffmpeg with AMR encoders not found: AMR conditions skipped")
    shard_dir = D2_DIR / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    sub = _subset()
    np.savez_compressed(D2_DIR / "subset.npz", **{k: v for k, v in sub.items()})
    paths = [None] * len(sub["row_index"])
    want = {int(r): i for i, r in enumerate(sub["row_index"])}
    with open(GATE_MANIFEST, encoding="utf-8") as f:
        next(f)
        for r, line in enumerate(f):
            if r in want:
                paths[want[r]] = line.split("\t")[0]
    assert all(paths), "subset rows missing from the gate manifest"
    n = len(paths)
    n_shards = (n + SHARD - 1) // SHARD
    total = n_shards * len(conditions)
    ctx.log(f"subset: {n} rows ({int((sub['label'] == 1).sum())} bonafide), "
            f"{n_shards} shards x {conditions}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(INCUMBENT), map_location="cpu", weights_only=False)
    assert_frozen_checkpoint(ckpt, "gpu_pipeline.d2")
    wavlm, dtype = load_frontend_for(ckpt, device)
    head = AASISTLHead(in_dim=768 if ckpt.get("frontend") == "microsoft/wavlm-base-plus" else 1024)
    head.load_state_dict(ckpt["state_dict"], strict=True)
    head.to(device).eval()
    layer = ckpt.get("layer_index")
    layer = -1 if layer is None else int(layer)

    unreadable: set = set()

    def raw16(p: str) -> np.ndarray:
        """The whole utterance as int16, as a phone network would carry it.
        The codec runs on this; the model's own window/normalisation
        (prepare_audio_window: crop to 4 s, zero-mean unit-variance) runs
        after it, exactly as in Phase 2. Normalising first and then
        quantising clipped ~20 % of samples, so the order matters."""
        try:
            x, sr = sf.read(p, dtype="float32")
        except Exception:
            # Counted and excluded from every metric, never scored as if real
            # (the corpus has a few unreadable files).
            unreadable.add(p)
            return np.zeros(64000, dtype=np.int16)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly

            x = resample_poly(x, 16000, sr).astype(np.float32)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16)

    def model_window(x16: np.ndarray) -> np.ndarray:
        return np.asarray(prepare_audio_window(x16.astype(np.float32) / 32767.0, sr=16000,
                                               target_duration=4.0, random_crop=False), dtype=np.float32)

    def degrade(x16: np.ndarray, cond: str) -> np.ndarray:
        if cond == "clean":
            return x16
        if cond == "g711u":
            return _g711u(x16)
        return _ffmpeg_roundtrip(ffmpeg, x16, cond)

    done = sum(1 for c in conditions for k in range(n_shards) if (shard_dir / f"{c}_{k:04d}.npz").exists())
    ctx.progress(done, total, "resuming" if done else "starting")
    t0, scored = time.time(), 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for k in range(n_shards):
            todo = [c for c in conditions if not (shard_dir / f"{c}_{k:04d}.npz").exists()]
            if not todo:
                continue
            idx = list(range(k * SHARD, min(n, (k + 1) * SHARD)))
            clean = list(ex.map(raw16, [paths[i] for i in idx]))
            for cond in todo:
                audio = list(ex.map(lambda x: model_window(degrade(x, cond)), clean))
                llr = []
                with torch.no_grad():
                    for b in range(0, len(audio), 64):
                        batch = np.stack(audio[b:b + 64])
                        hid = wavlm(torch.from_numpy(batch).to(dtype).to(device),
                                    output_hidden_states=True).hidden_states[layer].float()
                        _, logits = head(hid)
                        llr.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
                pcm_sha = hashlib.sha256(b"".join(a.tobytes() for a in audio)).hexdigest()
                tmp = shard_dir / f"{cond}_{k:04d}.tmp.npz"
                bad = np.array([paths[i] in unreadable for i in idx])
                np.savez(tmp, raw_llr=np.concatenate(llr), pos=np.array(idx), pcm_sha256=pcm_sha,
                         unreadable=bad)
                os.replace(tmp, shard_dir / f"{cond}_{k:04d}.npz")
                done += 1
                scored += len(idx)
                rate = scored / max(1e-6, time.time() - t0)
                ctx.progress(done, total, f"shard {k + 1}/{n_shards} {cond}  ({rate:.0f} utt/s)")
    return _d2_summarise(ctx, sub, conditions, n_shards)


def _d2_summarise(ctx: Ctx, sub: dict, conditions: List[str], n_shards: int) -> dict:
    tau = float(json.loads((T1_DIR / "decision_layer_calibration.json").read_text())
                ["chosen"]["operating_threshold"])
    label, attack = sub["label"], sub["attack"]
    res: dict = {"operating_threshold": tau, "subset_seed": SUBSET_SEED, "n": int(label.size),
                 "n_bonafide": int((label == 1).sum()), "per_attack_cap": PER_ATTACK,
                 "gate_manifest_sha256": sub["manifest_sha256"], "conditions": {}}
    n_all = int(sub["label"].size)
    for cond in conditions:
        llr = np.empty(n_all, dtype=np.float64)
        bad = np.zeros(n_all, dtype=bool)
        shas = []
        for k in range(n_shards):
            z = np.load(D2_DIR / "shards" / f"{cond}_{k:04d}.npz")
            llr[z["pos"]] = z["raw_llr"]
            if "unreadable" in z.files:
                bad[z["pos"]] = z["unreadable"]
            shas.append(str(z["pcm_sha256"]))
        res["unreadable_rows"] = max(res.get("unreadable_rows", 0), int(bad.sum()))
        keep = ~bad
        label, attack = sub["label"][keep], sub["attack"][keep]
        llr = llr[keep]
        spoof = label == 0
        miss = {a: round(100 * float(np.mean(llr[spoof & (attack == a)] >= tau)), 2)
                for a in sorted(set(attack[spoof]))}
        c = {"bonafide_frr_pct": round(100 * float(np.mean(llr[label == 1] < tau)), 2),
             "per_attack_miss_pct": miss,
             "worst_attack": max(miss, key=miss.get), "worst_miss_pct": max(miss.values()),
             "pcm_sha256": hashlib.sha256("".join(shas).encode()).hexdigest()}
        if cond == "clean":
            diff = np.abs(llr - sub["gate_raw_llr"][keep])
            c["parity_vs_phase2_dump"] = {"max_abs_diff": round(float(diff.max()), 5),
                                          "p99_abs_diff": round(float(np.percentile(diff, 99)), 5)}
        res["conditions"][cond] = c
    narrow = [c for c in ("g711u", "amrnb_12k2") if c in res["conditions"]]
    worst_nb = max((res["conditions"][c]["worst_miss_pct"] for c in narrow), default=None)
    res["stop_c"] = {
        "rule": "docs/13 §7: STOP-C if every candidate's worst narrowband miss is >= 50 %",
        "narrowband_conditions": narrow,
        "incumbent_worst_narrowband_miss_pct": worst_nb,
        "triggered": None if worst_nb is None else worst_nb >= 50.0,
    }
    (D2_DIR / "telephony_stopc.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    ctx.log(f"STOP-C: worst narrowband miss {worst_nb} % -> "
            f"{'TRIGGERED' if res['stop_c']['triggered'] else 'not triggered'}")
    return {"summary": str(D2_DIR / "telephony_stopc.json"), "stop_c": res["stop_c"]}


# --------------------------------------------------------------------------- W2.c Hindi ASR

W2C_DIR = OUT / "w2c"
W2C_PAIRS = 40
HI_WORDS = ["शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ"]
EN_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
W2C_VOICES = {"hi-IN-SwaraNeural": "hi", "hi-IN-MadhurNeural": "hi",
              "en-IN-NeerjaNeural": "en", "en-IN-PrabhatNeural": "en"}
W2C_ENGINES = [("tiny", "auto"), ("tiny", "hi,en"), ("small", "hi,en"),
               ("medium", "hi,en"), ("large-v3-turbo", "hi,en")]


def _tts_clip(ffmpeg: str, voice: str, text: str, path: Path) -> np.ndarray:
    """edge-tts (only the digit words go to the service) -> 16 kHz int16, cached."""
    if not path.exists():
        import asyncio

        import edge_tts

        mp3 = path.with_suffix(".mp3")
        asyncio.run(edge_tts.Communicate(text, voice, rate="-10%").save(str(mp3)))
        pcm = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(mp3),
                              "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
                             capture_output=True, check=True).stdout
        np.save(path, np.frombuffer(pcm, dtype=np.int16))
        mp3.unlink()
    return np.load(path)


def _pad(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lead = rng.normal(0, 30, int(0.6 * 16000)).astype(np.int16)
    tail = rng.normal(0, 30, int(0.8 * 16000)).astype(np.int16)
    return np.concatenate([lead, x.astype(np.int16), tail])


def w2c_hindi_asr(ctx: Ctx) -> dict:
    sys.path.insert(0, str(REPO / "vanirakshak-backend"))
    from vanirakshak.challenge.engine import extract_digits
    from vanirakshak.challenge.transcribe import ChallengeTranscriber
    from vanirakshak.data.synthetic import synthetic_spoofed

    ffmpeg = _ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is needed to decode TTS and apply codecs")
    W2C_DIR.mkdir(parents=True, exist_ok=True)
    (W2C_DIR / "tts").mkdir(exist_ok=True)
    rng = np.random.default_rng(SUBSET_SEED)
    pairs = ["".join(rng.choice(list("0123456789"), 2)) for _ in range(W2C_PAIRS)]

    ctx.progress(0, 1, "synthesising test answers (edge-tts)")
    clips = []  # (voice, lang, digits, audio)
    for voice, lang in W2C_VOICES.items():
        words = HI_WORDS if lang == "hi" else EN_WORDS
        for k, d in enumerate(pairs):
            x = _tts_clip(ffmpeg, voice, ", ".join(words[int(c)] for c in d),
                          W2C_DIR / "tts" / f"{voice}_{d}.npy")
            clips.append((voice, lang, d, _pad(x, k)))
    # Attacks: the right digits are never spoken.
    attacks = []
    for k, d in enumerate(pairs):
        wrong = "".join(str((int(c) + 1) % 10) for c in d)
        attacks.append(("tts_wrong_hi", d, _pad(_tts_clip(
            ffmpeg, "hi-IN-SwaraNeural", ", ".join(HI_WORDS[int(c)] for c in wrong),
            W2C_DIR / "tts" / f"hi-IN-SwaraNeural_{wrong}.npy"), k)))
        attacks.append(("noise", d, np.random.default_rng(k).normal(0, 3000, 4 * 16000)
                        .clip(-32767, 32767).astype(np.int16)))
        attacks.append(("vocoder", d, (np.clip(synthetic_spoofed(duration_sec=4.0, sr=16000,
                        rng=np.random.default_rng(k)), -1, 1) * 32767).astype(np.int16)))

    channels = {"clean": lambda x: x, "g711u": _g711u,
                "amrnb_12k2": lambda x: _ffmpeg_roundtrip(ffmpeg, x, "amrnb_12k2")}
    total = len(W2C_ENGINES)
    engines = {}
    for i, (size, langs) in enumerate(W2C_ENGINES):
        key = f"{size}/{langs}"
        cache = W2C_DIR / f"engine_v2_{size}_{langs.replace(',', '-')}.json"
        if cache.exists():
            engines[key] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        ctx.progress(i, total, f"loading whisper {key} (downloads on first use)")
        tr = ChallengeTranscriber(provider="local", model_size=size,
                                  languages=[] if langs == "auto" else langs.split(","), device="cuda")
        if not tr.available:
            engines[key] = {"error": tr.load_error}
            continue
        per: dict = {}
        per_voice: dict = {}
        fails: list = []
        readings: list = []  # raw engine output, so the parser can be re-graded offline
        t0, n = time.time(), 0
        for ch_name, ch in channels.items():
            for voice, lang, d, x in clips:
                # Graded as production grades: any reading with the digits.
                cands, _ = tr.transcribe_candidates(ch(x), 16000)
                ok = any(extract_digits(c)[:2] == list(d) for c in (cands or []))
                n += 1
                for cell in (per.setdefault(f"{lang}/{ch_name}", [0, 0]),
                             per_voice.setdefault(voice, [0, 0])):
                    cell[0] += ok
                    cell[1] += 1
                readings.append({"voice": voice, "lang": lang, "channel": ch_name,
                                 "digits": d, "readings": cands or []})
                if not ok and len(fails) < 12:
                    fails.append({"voice": voice, "channel": ch_name, "digits": d,
                                  "heard": " | ".join(cands or [])})
            ctx.progress(i, total, f"{key}: {ch_name} done")
        fa = {}
        for kind, d, x in attacks:
            cands, _ = tr.transcribe_candidates(x, 16000)
            fa.setdefault(kind, [0, 0])
            fa[kind][0] += any(extract_digits(c)[:2] == list(d) for c in (cands or []))
            fa[kind][1] += 1
        pct = lambda a, b: round(100 * a / max(1, b), 1)
        hi = [v for k, v in per.items() if k.startswith("hi/")]
        en = [v for k, v in per.items() if k.startswith("en/")]
        engines[key] = {
            "label": tr.label,
            "hi_pass_pct": pct(sum(a for a, _ in hi), sum(b for _, b in hi)),
            "en_pass_pct": pct(sum(a for a, _ in en), sum(b for _, b in en)),
            "per_cell_pass_pct": {k: pct(*v) for k, v in per.items()},
            "per_voice_pass_pct": {k: pct(*v) for k, v in per_voice.items()},
            "attack_false_accept_pct": {k: pct(*v) for k, v in fa.items()},
            "ms_per_answer": round(1000 * (time.time() - t0) / max(1, n), 1),
            "failed_examples": fails,
            "readings": readings,
        }
        cache.write_text(json.dumps(engines[key], indent=2, ensure_ascii=False), encoding="utf-8")
        ctx.log(f"{key}: hi {engines[key]['hi_pass_pct']} %  en {engines[key]['en_pass_pct']} %  "
                f"FA {engines[key]['attack_false_accept_pct']}  {engines[key]['ms_per_answer']} ms")
        del tr
    out = {"what": "Challenge ASR on TTS answers (edge-tts hi-IN / en-IN neural voices); "
                   "a proxy for real callers, not a friction measurement",
           "method": "one decode per language in the engine key with that language forced; "
                     "pass if any reading holds the two digits in order. The Devanagari "
                     "normaliser was designed from hi-IN-SwaraNeural failures; "
                     "hi-IN-MadhurNeural was not inspected (held-out voice).",
           "pairs": W2C_PAIRS, "voices": W2C_VOICES, "channels": list(channels),
           "engines": engines, "date": time.strftime("%Y-%m-%d")}
    (W2C_DIR / "hindi_asr.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    ctx.progress(total, total, "done")
    return {"summary": str(W2C_DIR / "hindi_asr.json")}


# --------------------------------------------------------------------------- G1: every head on telephony

G1_DIR = OUT / "g1_heads"
DEV_MANIFEST = Path("A:/cm-corpus/manifests/dev.tsv")
DEV_BONAFIDE = 4000
# Distinct weights only: *_cal and cm_aasist_op share weights with their base.
G1_HEADS = ["cm_aasist", "pre_aug_cm_aasist", "cm_aasist_layer5", "cm_aasist_attm",
            "cm_aasist_attm_uniform", "cm_aasist_meanpool",
            "cm_docs20_armb_s42", "cm_docs20_armb_s43", "cm_docs20_armb_s44",
            "cm_docs21_pct025_s42", "cm_docs21_pct025_s43",
            "cm_docs21_pct050_s42", "cm_docs21_pct050_s43",
            "cm_g4_control_s42", "cm_g4_control_s43",
            "cm_g4_rawboost_s42", "cm_g4_rawboost_s43", "cm_aasist_ft4_s42"]


def _dev_bonafide_paths() -> List[str]:
    rows = []
    with open(DEV_MANIFEST, encoding="utf-8") as f:
        next(f)
        for line in f:
            p, lab = line.rstrip("\n").split("\t")[:2]
            if lab == "bonafide":
                rows.append(p)
    rng = np.random.default_rng(SUBSET_SEED)
    pick = np.sort(rng.choice(len(rows), min(DEV_BONAFIDE, len(rows)), replace=False))
    return [rows[i] for i in pick]


def _raw16_reader(sf, unreadable: set):
    def raw16(p: str) -> np.ndarray:
        try:
            x, sr = sf.read(p, dtype="float32")
        except Exception:
            unreadable.add(p)
            return np.zeros(64000, dtype=np.int16)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16)
    return raw16


def _degrade(ffmpeg: str, x16: np.ndarray, cond: str) -> np.ndarray:
    if cond == "clean":
        return x16
    if cond == "g711u":
        return _g711u(x16)
    return _ffmpeg_roundtrip(ffmpeg, x16, cond)


def g1_telephony_heads(ctx: Ctx) -> dict:
    return _score_heads(ctx, G1_HEADS, G1_DIR)


def _score_heads(ctx: Ctx, head_names: List[str], out_dir: Path) -> dict:
    """Score checkpoints (by name under data/) on the STOP-C telephony
    protocol: shared by G1 (existing heads) and docs/16 (new heads)."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import soundfile as sf
    import torch

    from cm_frontend import is_finetuned, load_frontend_for
    from ml.aasist_model import AASISTLHead
    from ml.augment import prepare_audio_window
    from ml.train_cm import select_wavlm_features

    ffmpeg = _ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg with AMR encoders is required")
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Frozen heads share one WavLM forward per batch; fine-tuned ones get their own.
    heads, skipped, groups = {}, {}, {}
    for name in head_names:
        path = REPO / "data" / f"{name}.pt"
        if not path.exists():
            skipped[name] = "missing"
            continue
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        merge = ckpt.get("merge_layers") or None
        in_dim = 768 if ckpt.get("frontend", "microsoft/wavlm-base-plus") == "microsoft/wavlm-base-plus" else 1024
        head = AASISTLHead(in_dim=in_dim, merge_layers=merge)
        try:
            head.load_state_dict(ckpt["state_dict"], strict=True)
        except Exception as e:
            skipped[name] = f"does not load into AASISTLHead: {type(e).__name__}"
            continue
        li = ckpt.get("layer_index")
        heads[name] = (head.to(device).eval(), -1 if li is None else int(li), merge)
        key = name if is_finetuned(ckpt) else "frozen"
        groups.setdefault(key, (ckpt, []))[1].append(name)
    ctx.log(f"heads: {len(heads)} loaded; skipped {skipped}; frontends: {list(groups)}")
    frontends = {k: load_frontend_for(ck, device) for k, (ck, _) in groups.items()}

    sub = _subset()
    np.savez_compressed(out_dir / "subset.npz", **{k: v for k, v in sub.items()})
    want = {int(r): i for i, r in enumerate(sub["row_index"])}
    gate_paths: List[Optional[str]] = [None] * len(want)
    with open(GATE_MANIFEST, encoding="utf-8") as f:
        next(f)
        for r, line in enumerate(f):
            if r in want:
                gate_paths[want[r]] = line.split("\t")[0]
    dev_paths = _dev_bonafide_paths()
    sets = {"dev_clean": (dev_paths, ["clean"]), "gate": (gate_paths, CONDITIONS)}

    unreadable: set = set()
    raw16 = _raw16_reader(sf, unreadable)

    def model_window(x16: np.ndarray) -> np.ndarray:
        w = prepare_audio_window(x16.astype(np.float32) / 32767.0, sr=16000,
                                 target_duration=4.0, random_crop=False)
        return np.asarray(w, dtype=np.float32)

    jobs = [(s, cond, k) for s, (paths, conds) in sets.items() for cond in conds
            for k in range((len(paths) + SHARD - 1) // SHARD)]
    done = sum((shard_dir / f"{s}_{c}_{k:04d}.npz").exists() for s, c, k in jobs)
    ctx.progress(done, len(jobs), "resuming" if done else "starting")
    t0, n_scored = time.time(), 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for s, cond, k in jobs:
            out = shard_dir / f"{s}_{cond}_{k:04d}.npz"
            if out.exists():
                continue
            paths = sets[s][0]
            idx = list(range(k * SHARD, min(len(paths), (k + 1) * SHARD)))
            audio = list(ex.map(lambda p: model_window(_degrade(ffmpeg, raw16(p), cond)),
                                [paths[i] for i in idx]))
            scores: Dict[str, list] = {name: [] for name in heads}
            with torch.no_grad():
                for b in range(0, len(audio), 32):
                    batch = np.stack(audio[b:b + 32])
                    for key, (_, names) in groups.items():
                        wavlm, dtype = frontends[key]
                        hs = wavlm(torch.from_numpy(batch).to(dtype).to(device), output_hidden_states=True)
                        for name in names:
                            head, li, merge = heads[name]
                            _, logits = head(select_wavlm_features(hs, li, merge))
                            scores[name].append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
            bad = np.array([paths[i] in unreadable for i in idx])
            tmp = shard_dir / f"{s}_{cond}_{k:04d}.tmp.npz"
            np.savez(tmp, pos=np.array(idx), unreadable=bad,
                     **{f"llr__{n}": np.concatenate(v) for n, v in scores.items()})
            os.replace(tmp, out)
            done += 1
            n_scored += len(idx)
            ctx.progress(done, len(jobs),
                         f"{s} {cond} shard {k + 1}  ({n_scored / (time.time() - t0):.0f} utt/s)")
    return _g1_summarise(ctx, sub, len(dev_paths), list(heads), skipped, out_dir)


def _g1_load(set_name: str, cond: str, n: int, name: str, out_dir: Path = G1_DIR):
    llr = np.full(n, np.nan)
    bad = np.zeros(n, dtype=bool)
    for f in sorted((out_dir / "shards").glob(f"{set_name}_{cond}_*.npz")):
        z = np.load(f)
        llr[z["pos"]] = z[f"llr__{name}"]
        bad[z["pos"]] = z["unreadable"]
    return llr, bad


def _g1_summarise(ctx: Ctx, sub: dict, n_dev: int, names: List[str], skipped: dict,
                  out_dir: Path = G1_DIR) -> dict:
    label, attack = sub["label"], sub["attack"]
    n = label.size
    res: dict = {"what": "Every distinct CM checkpoint on the STOP-C telephony protocol (docs/13 §12)",
                 "threshold": "per head: 5th percentile of its own raw LLR on a seeded dev-bonafide set",
                 "n_gate": int(n), "n_dev_bonafide": n_dev, "skipped": skipped, "heads": {}}
    for name in names:
        dev, dbad = _g1_load("dev_clean", "clean", n_dev, name, out_dir)
        tau = float(np.percentile(dev[~dbad], 5.0))
        h: dict = {"operating_threshold": round(tau, 4), "conditions": {}}
        for cond in CONDITIONS:
            llr, bad = _g1_load("gate", cond, n, name, out_dir)
            keep = ~bad
            lab, atk, s = label[keep], attack[keep], llr[keep]
            spoof = lab == 0
            miss = {a: round(100 * float(np.mean(s[spoof & (atk == a)] >= tau)), 2)
                    for a in sorted(set(atk[spoof]))}
            h["conditions"][cond] = {
                "bonafide_frr_pct": round(100 * float(np.mean(s[lab == 1] < tau)), 2),
                "worst_attack": max(miss, key=miss.get),
                "worst_miss_pct": max(miss.values()),
                "mean_miss_pct": round(float(np.mean(list(miss.values()))), 2),
                "per_attack_miss_pct": miss,
            }
        nb = max(h["conditions"][c]["worst_miss_pct"] for c in ("g711u", "amrnb_12k2"))
        h["worst_narrowband_miss_pct"] = nb
        h["passes_stop_c_bar"] = nb < 50.0
        res["heads"][name] = h
    ranking = sorted(res["heads"], key=lambda k: res["heads"][k]["worst_narrowband_miss_pct"])
    res["ranking_by_worst_narrowband_miss"] = [
        {"head": k,
         "worst_narrowband_miss_pct": res["heads"][k]["worst_narrowband_miss_pct"],
         "narrowband_mean_miss_pct": round(float(np.mean(
             [res["heads"][k]["conditions"][c]["mean_miss_pct"] for c in ("g711u", "amrnb_12k2")])), 2),
         "clean_worst_miss_pct": res["heads"][k]["conditions"]["clean"]["worst_miss_pct"],
         "clean_frr_pct": res["heads"][k]["conditions"]["clean"]["bonafide_frr_pct"]}
        for k in ranking]
    (out_dir / "heads_telephony.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in res["ranking_by_worst_narrowband_miss"][:6]:
        ctx.log(f"{r['head']:24s} worst NB miss {r['worst_narrowband_miss_pct']:6.2f} %  "
                f"NB mean {r['narrowband_mean_miss_pct']:6.2f} %  clean worst {r['clean_worst_miss_pct']:6.2f} %")
    return {"summary": str(out_dir / "heads_telephony.json"), "best": ranking[0],
            "any_passes": any(res["heads"][k]["passes_stop_c_bar"] for k in ranking)}


# --------------------------------------------------------------------------- docs/16 (G2)

G2_DIR = OUT / "g2"
G2_COMMON = ["--merge-layers", "12", "--layer-index", "5", "--epochs", "10", "--batch-size", "64",
             "--lr", "5e-4", "--weight-decay", "1e-4", "--patience", "5", "--max-dev-utts", "30000",
             "--streaming-dev", "--workers", "24", "--p-codec", "0", "--p-noise", "0",
             "--p-bandpass", "0", "--p-rir", "0", "--p-gain", "0"]
G2_ARMS = {"C": ["--real-codec-p", "0"], "R": ["--real-codec-p", "0.5"]}


def _g2_name(arm: str, seed: int) -> str:
    return f"cm_docs16_{arm}_s{seed}"


def _g2_train(ctx: Ctx, arm: str, seed: int) -> dict:
    """One docs/16 run. A run is either complete (a .done marker next to the
    checkpoint) or re-run from scratch (docs/16 §6: never resumed)."""
    ckpt = REPO / "data" / f"{_g2_name(arm, seed)}.pt"
    done = ckpt.with_suffix(".done.json")
    if ckpt.exists() and done.exists():
        ctx.log(f"{ckpt.name}: already complete")
        return json.loads(done.read_text(encoding="utf-8"))
    if ckpt.exists():
        ckpt.unlink()  # partial run: docs/16 §6 says re-run, not resume
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    ctx.log(f"training {ckpt.name} at git {head}")
    t0 = time.time()
    p = ctx.run([PY, "ml/train_cm.py", *G2_COMMON, *G2_ARMS[arm], "--seed", str(seed),
                 "--output", str(ckpt)], timeout=12 * 3600)
    if p.returncode != 0 or not ckpt.exists():
        raise RuntimeError(f"training {ckpt.name} failed; see log")
    rec = {"checkpoint": str(ckpt), "arm": arm, "seed": seed, "git_head": head,
           "wall_hours": round((time.time() - t0) / 3600, 2)}
    done.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def _g2_futility() -> Optional[dict]:
    f = G2_DIR / "futility.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def g2_train_s42(ctx: Ctx) -> dict:
    out = {}
    for i, arm in enumerate(("C", "R")):
        ctx.progress(i, 2, f"training arm {arm} seed 42")
        out[arm] = _g2_train(ctx, arm, 42)
    ctx.progress(2, 2, "seed 42 trained")
    return out


def g2_futility(ctx: Ctx) -> dict:
    """docs/16 §5: stop if R42 is not at least 1 pp better than C42."""
    names = [_g2_name("C", 42), _g2_name("R", 42)]
    res = _score_heads(ctx, names, G2_DIR / "s42")
    heads = json.loads((G2_DIR / "s42" / "heads_telephony.json").read_text(encoding="utf-8"))["heads"]
    c, r = (heads[n]["worst_narrowband_miss_pct"] for n in names)
    stop = r > c - 1.0
    rec = {"rule": "docs/16 §5: stop if worst-NB miss R42 > C42 - 1.0 pp",
           "C42_worst_nb_miss_pct": c, "R42_worst_nb_miss_pct": r, "stop": stop,
           "verdict_if_stop": "NO_EFFECT (early)"}
    G2_DIR.mkdir(parents=True, exist_ok=True)
    (G2_DIR / "futility.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    ctx.log(f"futility: C42 {c} %  R42 {r} %  -> {'STOP' if stop else 'continue'}")
    return {**rec, "scoring": res}


def g2_train_rest(ctx: Ctx) -> dict:
    fut = _g2_futility()
    if fut is None:
        raise RuntimeError("g2_futility has not run")
    if fut["stop"]:
        ctx.log("cancelled by the docs/16 §5 futility rule")
        return {"cancelled": True, "futility": fut}
    runs = [(a, s) for s in (43, 44) for a in ("C", "R")]
    out = {}
    for i, (arm, seed) in enumerate(runs):
        ctx.progress(i, len(runs), f"training arm {arm} seed {seed}")
        out[f"{arm}{seed}"] = _g2_train(ctx, arm, seed)
    ctx.progress(len(runs), len(runs), "all seeds trained")
    return out


def g2_eval(ctx: Ctx) -> dict:
    fut = _g2_futility()
    seeds = (42,) if fut and fut["stop"] else (42, 43, 44)
    names = [_g2_name(a, s) for s in seeds for a in ("C", "R")]
    return _score_heads(ctx, names, G2_DIR / "all")


def g2_adjudicate(ctx: Ctx) -> dict:
    fut = _g2_futility()
    heads = json.loads((G2_DIR / "all" / "heads_telephony.json").read_text(encoding="utf-8"))["heads"]

    def vals(arm: str, key) -> List[float]:
        return [key(heads[n]) for n in heads if n.startswith(f"cm_docs16_{arm}_")]

    nb = lambda h: h["worst_narrowband_miss_pct"]
    cw = lambda h: h["conditions"]["clean"]["worst_miss_pct"]
    cf = lambda h: h["conditions"]["clean"]["bonafide_frr_pct"]
    C, R = vals("C", nb), vals("R", nb)
    mean = lambda x: float(np.mean(x))
    p1 = mean(R) <= mean(C) - 5.0 and max(R) < min(C)
    p2 = mean(R) < 50.0
    p3 = mean(vals("R", cw)) <= mean(vals("C", cw)) + 3.0 and mean(vals("R", cf)) <= mean(vals("C", cf)) + 2.0
    if fut and fut["stop"]:
        verdict = "NO_EFFECT (early, docs/16 §5)"
    elif not p3:
        verdict = "HARMS"
    elif p1 and p2:
        verdict = "ADOPT"
    elif p1:
        verdict = "IMPROVES_NOT_ENOUGH"
    else:
        verdict = "NO_EFFECT"
    best_r = min((n for n in heads if n.startswith("cm_docs16_R_")), key=lambda n: nb(heads[n]))
    rec = {"plan": "docs/16", "verdict": verdict,
           "P16_1_effect": p1, "P16_2_stop_c_bar": p2, "P16_3_no_clean_regression": p3,
           "worst_nb_miss_pct": {"C": C, "R": R, "C_mean": round(mean(C), 2), "R_mean": round(mean(R), 2)},
           "clean_worst_miss_pct": {"C": vals("C", cw), "R": vals("R", cw)},
           "clean_frr_pct": {"C": vals("C", cf), "R": vals("R", cf)},
           "best_R_seed": best_r if verdict == "ADOPT" else None,
           "futility": fut, "date": time.strftime("%Y-%m-%d")}
    (G2_DIR / "verdict.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    ctx.log(f"docs/16 verdict: {verdict}  (worst NB miss C {rec['worst_nb_miss_pct']['C_mean']} % "
            f"vs R {rec['worst_nb_miss_pct']['R_mean']} %)")
    return rec


# --------------------------------------------------------------------------- G1: speaker-grouped T1

SPK_DIR = OUT / "g1_spk"
PROTOCOLS = Path("A:/cm-corpus/protocols")


def g1_speaker_grouped(ctx: Ctx) -> dict:
    """docs/14 W1.b: re-measure T1's chosen thresholds with every simulated
    call drawn from ONE speaker (ASVspoof 5 protocol speaker ids) instead of
    i.i.d. windows. Recorded, not a re-grade of docs/13 §11."""
    SPK_DIR.mkdir(parents=True, exist_ok=True)
    out = SPK_DIR / "t1_speaker_grouped.json"
    if not out.exists():
        ctx.progress(0, 1, "simulating speaker-grouped calls (CPU)")
        p = ctx.run([PY, "scripts/calibrate_decision_layer.py",
                     "--dev-scores", str(DEV_DUMP), "--eval-scores", str(GATE_DUMP),
                     "--speaker-grouped",
                     "--dev-manifest", "A:/cm-corpus/manifests/dev.tsv",
                     "--dev-protocol", str(PROTOCOLS / "ASVspoof5.dev.track_1.tsv"),
                     "--eval-manifest", str(GATE_MANIFEST),
                     "--eval-protocol", str(PROTOCOLS / "ASVspoof5.eval.track_1.tsv"),
                     "--fixed-from", str(T1_DIR / "decision_layer_calibration.json"),
                     "--challenge-consecutive", "3", "--out", str(out)], timeout=6 * 3600)
        if p.returncode != 0:
            raise RuntimeError("speaker-grouped calibration failed; see log")
    grouped = json.loads(out.read_text(encoding="utf-8"))["results"]["eval"]
    iid = json.loads((T1_DIR / "decision_layer_calibration.json").read_text(encoding="utf-8"))["results"]["eval"]
    cmp: dict = {}
    for cell in grouped["after"]:
        g, i = grouped["after"][cell], iid["after"][cell]
        cmp[cell] = {
            "genuine_block_by_end": {"iid": i["genuine"]["p_block_by_end"], "grouped": g["genuine"]["p_block_by_end"]},
            "genuine_challenge_60s": {"iid": i["genuine"]["p_challenge_within_60s"],
                                      "grouped": g["genuine"]["p_challenge_within_60s"]},
            "spoof_block_worst_attack": {"iid": _worst_attack(i), "grouped": _worst_attack(g)},
        }
    summary = {
        "what": "T1 thresholds (docs/13 §11) re-measured with speaker-grouped calls on the gate split",
        "cells": cmp,
        "max_genuine_block": {"iid": max(v["genuine_block_by_end"]["iid"] for v in cmp.values()),
                              "grouped": max(v["genuine_block_by_end"]["grouped"] for v in cmp.values())},
        "min_spoof_block_worst_attack": {"iid": min(v["spoof_block_worst_attack"]["iid"] for v in cmp.values()),
                                         "grouped": min(v["spoof_block_worst_attack"]["grouped"] for v in cmp.values())},
    }
    (SPK_DIR / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ctx.log(f"genuine BLOCK max: iid {summary['max_genuine_block']['iid']}  grouped "
            f"{summary['max_genuine_block']['grouped']};  worst spoof BLOCK: iid "
            f"{summary['min_spoof_block_worst_attack']['iid']}  grouped {summary['min_spoof_block_worst_attack']['grouped']}")
    ctx.progress(1, 1, "done")
    return {"summary": str(SPK_DIR / "comparison.json"), **{k: summary[k] for k in
            ("max_genuine_block", "min_spoof_block_worst_attack")}}


# --------------------------------------------------------------------------- driver

STAGES: Dict[str, Callable[[Ctx], dict]] = {
    "t1_calibrate": t1_calibrate,
    "t1_grade": t1_grade,
    "t2_cuda": t2_cuda,
    "t2_cpu": t2_cpu,
    "d2_telephony": d2_telephony,
    "w2c_hindi_asr": w2c_hindi_asr,
    "g1_telephony_heads": g1_telephony_heads,
    "g2_train_s42": g2_train_s42,
    "g2_futility": g2_futility,
    "g2_train_rest": g2_train_rest,
    "g2_eval": g2_eval,
    "g2_adjudicate": g2_adjudicate,
    "g1_speaker_grouped": g1_speaker_grouped,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=OUT / "pipeline")
    ap.add_argument("--state-name", default="state.json",
                    help="a separate state file lets a second runner work alongside the first")
    ap.add_argument("--only", action="append", choices=list(STAGES))
    ap.add_argument("--redo", action="append", choices=list(STAGES), default=[],
                    help="re-run a stage even if it is marked done")
    args = ap.parse_args()
    state = State(args.root, args.state_name)
    for name in STAGES:
        if not args.only or name in args.only:
            state.stage(name)
    state.data["pid"] = os.getpid()
    state.save()
    failed = False
    for name, fn in STAGES.items():
        if args.only and name not in args.only:
            continue
        st = state.stage(name)
        if st.get("status") == "done" and name not in args.redo:
            print(f"{name}: done, skipping")
            continue
        state.update(name, status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"), error=None)
        ctx = Ctx(state, name)
        try:
            result = fn(ctx)
            state.update(name, status="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), result=result)
        except Exception as e:
            ctx.log(traceback.format_exc())
            state.update(name, status="failed", error=f"{type(e).__name__}: {e}",
                         finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
            failed = True
    state.data["pid"] = None
    state.save()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
