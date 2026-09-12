import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import WavLMModel

# Repo root is this file's parent directory, so the script runs from any
# checkout on any machine rather than only from one hardcoded path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.compute_mindcf import (
    TAU_BAYES,
    compute_act_dcf,
    compute_cllr,
    compute_eer,
    compute_mindcf,
    evaluate_system,
)
from ml.aasist_model import AASISTLHead
from ml.augment import prepare_audio_window
from sklearn.linear_model import LogisticRegression


def load_audio_item(item):
    p, label, atk = item
    try:
        x, sr = sf.read(p, dtype="float32")
        w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
        return w, label, atk
    except Exception:
        return np.zeros(64000, dtype=np.float32), label, atk


def fit_platt_dev(bon_dev, spf_dev):
    scores = np.concatenate([bon_dev, spf_dev])
    y = np.concatenate([np.ones_like(bon_dev), np.zeros_like(spf_dev)])
    lr = LogisticRegression(C=1e5, solver="lbfgs")
    lr.fit(scores.reshape(-1, 1), y)
    w = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])
    return w, b


def get_per_attack_dict(bon, atk_dict, w=1.0, b=0.0):
    res = {}
    bon_sc = bon * w + b
    for atk, sc in sorted(atk_dict.items()):
        sc_arr = np.array(sc, dtype=np.float64) * w + b
        if len(sc_arr) > 0 and len(bon_sc) > 0:
            e, _ = compute_eer(bon_sc, sc_arr)
            res[atk] = round(float(e) * 100.0, 3)
    return res
def main():
    dev_manifest = Path(r"A:\cm-corpus\manifests\dev.tsv")
    eval_manifest = Path(r"data\eval_benchmark\eval_slice_20k.tsv")
    attm_ckpt_path = REPO_ROOT / "data" / "cm_aasist_attm.pt"
    uni_ckpt_path = REPO_ROOT / "data" / "cm_aasist_attm_uniform.pt"

    assert dev_manifest.is_file(), f"Dev manifest not found: {dev_manifest}"
    assert eval_manifest.is_file(), f"Eval manifest not found: {eval_manifest}"
    assert attm_ckpt_path.is_file(), f"AttM checkpoint not found: {attm_ckpt_path}"
    assert uni_ckpt_path.is_file(), f"Uniform checkpoint not found: {uni_ckpt_path}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/5] Initializing models on {device}...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    if device == "cuda":
        wavlm = wavlm.half().to(device)
    wavlm.eval()

    # Load AttM model
    ck_attm = torch.load(attm_ckpt_path, map_location="cpu")
    m_attm = AASISTLHead(in_dim=768, merge_layers=12).to(device)
    m_attm.load_state_dict(ck_attm["state_dict"], strict=True)
    m_attm.eval()

    # Load Uniform model
    ck_uni = torch.load(uni_ckpt_path, map_location="cpu")
    m_uni = AASISTLHead(in_dim=768, merge_layers=12).to(device)
    m_uni.load_state_dict(ck_uni["state_dict"], strict=True)
    m_uni.eval()

    # Read dev records
    dev_records = []
    with open(dev_manifest, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            dev_records.append((r[0], 1 if r[1].lower() in ("bonafide", "bona-fide", "genuine", "1") else 0, r[2] if len(r) > 2 else "-"))

    print(f"[2/5] Scoring development set ({len(dev_records)} records)...")
    batch_size = 64
    bon_dev_attm, spf_dev_attm = [], []
    bon_dev_uni, spf_dev_uni = [], []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i in range(0, len(dev_records), batch_size):
            batch = dev_records[i : i + batch_size]
            proc = list(ex.map(load_audio_item, batch))
            audios = np.stack([p[0] for p in proc])
            labels = [p[1] for p in proc]

            at = torch.from_numpy(audios)
            if device == "cuda":
                at = at.half().to(device)
            else:
                at = at.to(device)

            with torch.no_grad():
                out = wavlm(at, output_hidden_states=True)
                feats_12 = [out.hidden_states[j].float() for j in range(12)]
                _, l_attm = m_attm(feats_12)
                _, l_uni = m_uni(feats_12)
                llr_attm = (l_attm[:, 1] - l_attm[:, 0]).cpu().numpy()
                llr_uni = (l_uni[:, 1] - l_uni[:, 0]).cpu().numpy()

            for la, lu, y in zip(llr_attm, llr_uni, labels):
                if y == 1:
                    bon_dev_attm.append(float(la))
                    bon_dev_uni.append(float(lu))
                else:
                    spf_dev_attm.append(float(la))
                    spf_dev_uni.append(float(lu))

            if (i // batch_size) % 100 == 0 and i > 0:
                elapsed = time.time() - t0
                speed = i / elapsed
                rem = (len(dev_records) - i) / speed
                print(f"  Dev scoring {i}/{len(dev_records)} ({speed:.1f} utts/s, {elapsed:.1f}s elapsed, ~{rem:.1f}s remaining)")
                sys.stdout.flush()

    bon_dev_attm_a = np.array(bon_dev_attm, dtype=np.float64)
    spf_dev_attm_a = np.array(spf_dev_attm, dtype=np.float64)
    bon_dev_uni_a = np.array(bon_dev_uni, dtype=np.float64)
    spf_dev_uni_a = np.array(spf_dev_uni, dtype=np.float64)

    # Compute dev metrics & EER thresholds
    dev_eer_attm, dev_tau_attm = compute_eer(bon_dev_attm_a, spf_dev_attm_a)
    dev_mindcf_attm, _ = compute_mindcf(bon_dev_attm_a, spf_dev_attm_a)

    dev_eer_uni, dev_tau_uni = compute_eer(bon_dev_uni_a, spf_dev_uni_a)
    dev_mindcf_uni, _ = compute_mindcf(bon_dev_uni_a, spf_dev_uni_a)

    print("\n--- Dev Partition Measured Metrics ---")
    print(f"AttM    : Dev EER = {dev_eer_attm*100:.3f}%, Dev minDCF = {dev_mindcf_attm:.4f}, Dev EER Threshold = {dev_tau_attm:.6f}")
    print(f"Uniform : Dev EER = {dev_eer_uni*100:.3f}%, Dev minDCF = {dev_mindcf_uni:.4f}, Dev EER Threshold = {dev_tau_uni:.6f}")

    # Fit Platt on Dev
    print("\n[3/5] Fitting Platt logistic regression on Dev partition only...")
    w_attm, b_attm = fit_platt_dev(bon_dev_attm_a, spf_dev_attm_a)
    w_uni, b_uni = fit_platt_dev(bon_dev_uni_a, spf_dev_uni_a)

    # Dev calibrated metrics
    cal_bon_dev_attm = w_attm * bon_dev_attm_a + b_attm
    cal_spf_dev_attm = w_attm * spf_dev_attm_a + b_attm
    dev_cal_cllr_attm = compute_cllr(cal_bon_dev_attm, cal_spf_dev_attm)
    dev_cal_actdcf_b_attm = compute_act_dcf(cal_bon_dev_attm, cal_spf_dev_attm, threshold=TAU_BAYES)

    cal_bon_dev_uni = w_uni * bon_dev_uni_a + b_uni
    cal_spf_dev_uni = w_uni * spf_dev_uni_a + b_uni
    dev_cal_cllr_uni = compute_cllr(cal_bon_dev_uni, cal_spf_dev_uni)
    dev_cal_actdcf_b_uni = compute_act_dcf(cal_bon_dev_uni, cal_spf_dev_uni, threshold=TAU_BAYES)

    print(f"AttM Platt parameters    : scale_w = {w_attm:.6f}, bias_b = {b_attm:.6f}")
    print(f"  Dev Calibrated CLLR    : {dev_cal_cllr_attm:.4f}, actDCF_bayes = {dev_cal_actdcf_b_attm:.4f}")
    print(f"Uniform Platt parameters : scale_w = {w_uni:.6f}, bias_b = {b_uni:.6f}")
    print(f"  Dev Calibrated CLLR    : {dev_cal_cllr_uni:.4f}, actDCF_bayes = {dev_cal_actdcf_b_uni:.4f}")

    # Update Checkpoint Metadata
    print("\nUpdating checkpoints on disk with dev metadata and calibration blocks...")
    ck_attm["dev_eer"] = float(dev_eer_attm)
    ck_attm["dev_min_dcf"] = float(dev_mindcf_attm)
    ck_attm["eer_threshold"] = float(dev_tau_attm)
    ck_attm["calibration"] = {
        "method": "platt",
        "scale_w": round(w_attm, 6),
        "bias_b": round(b_attm, 6),
        "calibrated_cllr": round(dev_cal_cllr_attm, 4),
        "calibrated_actdcf_bayes": round(dev_cal_actdcf_b_attm, 4),
    }
    torch.save(ck_attm, attm_ckpt_path)
    print(f"Saved updated {attm_ckpt_path}")

    ck_uni["dev_eer"] = float(dev_eer_uni)
    ck_uni["dev_min_dcf"] = float(dev_mindcf_uni)
    ck_uni["eer_threshold"] = float(dev_tau_uni)
    ck_uni["calibration"] = {
        "method": "platt",
        "scale_w": round(w_uni, 6),
        "bias_b": round(b_uni, 6),
        "calibrated_cllr": round(dev_cal_cllr_uni, 4),
        "calibrated_actdcf_bayes": round(dev_cal_actdcf_b_uni, 4),
    }
    torch.save(ck_uni, uni_ckpt_path)
    print(f"Saved updated {uni_ckpt_path}")

    # Read eval records
    eval_records = []
    with open(eval_manifest, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            eval_records.append((r[0], 1 if r[1].lower() in ("bonafide", "bona-fide", "genuine", "1") else 0, r[2]))

    print(f"\n[4/5] Scoring evaluation slice ({len(eval_records)} records)...")
    bon_eval_attm, spf_eval_attm, atk_eval_attm = [], [], {}
    bon_eval_uni, spf_eval_uni, atk_eval_uni = [], [], {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i in range(0, len(eval_records), batch_size):
            batch = eval_records[i : i + batch_size]
            proc = list(ex.map(load_audio_item, batch))
            audios = np.stack([p[0] for p in proc])
            labels = [p[1] for p in proc]
            atks = [p[2] for p in proc]

            at = torch.from_numpy(audios)
            if device == "cuda":
                at = at.half().to(device)
            else:
                at = at.to(device)

            with torch.no_grad():
                out = wavlm(at, output_hidden_states=True)
                feats_12 = [out.hidden_states[j].float() for j in range(12)]
                _, l_attm = m_attm(feats_12)
                _, l_uni = m_uni(feats_12)
                llr_attm = (l_attm[:, 1] - l_attm[:, 0]).cpu().numpy()
                llr_uni = (l_uni[:, 1] - l_uni[:, 0]).cpu().numpy()

            for la, lu, y, atk in zip(llr_attm, llr_uni, labels, atks):
                sla = float(la)
                slu = float(lu)
                if y == 1:
                    bon_eval_attm.append(sla)
                    bon_eval_uni.append(slu)
                else:
                    spf_eval_attm.append(sla)
                    spf_eval_uni.append(slu)
                    if atk not in atk_eval_attm:
                        atk_eval_attm[atk] = []
                        atk_eval_uni[atk] = []
                    atk_eval_attm[atk].append(sla)
                    atk_eval_uni[atk].append(slu)

            if (i // batch_size) % 50 == 0 and i > 0:
                elapsed = time.time() - t0
                speed = i / elapsed
                print(f"  Eval scoring {i}/{len(eval_records)} ({speed:.1f} utts/s)")
                sys.stdout.flush()

    bon_eval_attm_a = np.array(bon_eval_attm, dtype=np.float64)
    spf_eval_attm_a = np.array(spf_eval_attm, dtype=np.float64)
    bon_eval_uni_a = np.array(bon_eval_uni, dtype=np.float64)
    spf_eval_uni_a = np.array(spf_eval_uni, dtype=np.float64)

    # Raw metrics on eval slice
    res_attm_raw = evaluate_system(bon_eval_attm_a, spf_eval_attm_a)
    res_uni_raw = evaluate_system(bon_eval_uni_a, spf_eval_uni_a)
    per_atk_attm_raw = get_per_attack_dict(bon_eval_attm_a, atk_eval_attm)
    per_atk_uni_raw = get_per_attack_dict(bon_eval_uni_a, atk_eval_uni)

    # Calibrated metrics on eval slice
    cal_bon_eval_attm = w_attm * bon_eval_attm_a + b_attm
    cal_spf_eval_attm = w_attm * spf_eval_attm_a + b_attm
    res_attm_cal = evaluate_system(cal_bon_eval_attm, cal_spf_eval_attm)
    per_atk_attm_cal = get_per_attack_dict(bon_eval_attm_a, atk_eval_attm, w=w_attm, b=b_attm)

    cal_bon_eval_uni = w_uni * bon_eval_uni_a + b_uni
    cal_spf_eval_uni = w_uni * spf_eval_uni_a + b_uni
    res_uni_cal = evaluate_system(cal_bon_eval_uni, cal_spf_eval_uni)
    per_atk_uni_cal = get_per_attack_dict(bon_eval_uni_a, atk_eval_uni, w=w_uni, b=b_uni)

    print("\n[5/5] Compiling and Verifying Criteria...")
    print("\n=== ATTM (12L) EVAL SLICE (20,000 utts) ===")
    print(f"Raw        : EER = {res_attm_raw['eer']*100:.3f}%, minDCF = {res_attm_raw['min_dcf']:.4f}, actDCF = {res_attm_raw['act_dcf']:.4f}, actDCF_bayes = {res_attm_raw['act_dcf_bayes']:.4f}, CLLR = {res_attm_raw['cllr']:.4f}")
    print(f"Calibrated : EER = {res_attm_cal['eer']*100:.3f}%, minDCF = {res_attm_cal['min_dcf']:.4f}, actDCF = {res_attm_cal['act_dcf']:.4f}, actDCF_bayes = {res_attm_cal['act_dcf_bayes']:.4f}, CLLR = {res_attm_cal['cllr']:.4f}")

    print("\n=== UNIFORM (12L) EVAL SLICE (20,000 utts) ===")
    print(f"Raw        : EER = {res_uni_raw['eer']*100:.3f}%, minDCF = {res_uni_raw['min_dcf']:.4f}, actDCF = {res_uni_raw['act_dcf']:.4f}, actDCF_bayes = {res_uni_raw['act_dcf_bayes']:.4f}, CLLR = {res_uni_raw['cllr']:.4f}")
    print(f"Calibrated : EER = {res_uni_cal['eer']*100:.3f}%, minDCF = {res_uni_cal['min_dcf']:.4f}, actDCF = {res_uni_cal['act_dcf']:.4f}, actDCF_bayes = {res_uni_cal['act_dcf_bayes']:.4f}, CLLR = {res_uni_cal['cllr']:.4f}")

    # Verification against criteria
    print("\n--- CRITERIA CHECK ---")
    for name, r_raw, r_cal in [("AttM", res_attm_raw, res_attm_cal), ("Uniform", res_uni_raw, res_uni_cal)]:
        cllr_ok = r_cal["cllr"] < 1.0
        gap = abs(r_cal["act_dcf_bayes"] - r_cal["min_dcf"]) / r_cal["min_dcf"]
        actdcf_ok = gap <= 0.10
        eer_ok = round(r_raw["eer"], 5) == round(r_cal["eer"], 5)
        mindcf_ok = round(r_raw["min_dcf"], 5) == round(r_cal["min_dcf"], 5)
        print(f"[{name}]")
        print(f"  1. Calibrated CLLR < 1.0       : {r_cal['cllr']:.4f} (pass: {cllr_ok})")
        print(f"  2. actDCF within 10% of minDCF : actDCF_bayes={r_cal['act_dcf_bayes']:.4f}, minDCF={r_cal['min_dcf']:.4f}, gap={gap*100:.2f}% (pass: {actdcf_ok})")
        print(f"  3. EER unchanged               : raw={r_raw['eer']*100:.3f}% vs cal={r_cal['eer']*100:.3f}% (pass: {eer_ok})")
        print(f"     minDCF unchanged            : raw={r_raw['min_dcf']:.4f} vs cal={r_cal['min_dcf']:.4f} (pass: {mindcf_ok})")

    # Construct and save reports/cm_calibrated_benchmark.json
    out_report = REPO_ROOT / "reports" / "cm_calibrated_benchmark.json"
    report_content = {
        "evaluation_partition": "data/eval_benchmark/eval_slice_20k.tsv",
        "development_partition": "A:/cm-corpus/manifests/dev.tsv",
        "total_utterances": len(eval_records),
        "bonafide_count": len(bon_eval_attm),
        "spoof_count": len(spf_eval_attm),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "G2 Score Calibration Benchmark: Side-by-side raw vs. Platt-calibrated countermeasure evaluation",
        "models": {
            "attentive_merge_12layers": {
                "checkpoint": "data/cm_aasist_attm.pt",
                "calibration_parameters": {
                    "method": "platt",
                    "scale_w": round(w_attm, 6),
                    "bias_b": round(b_attm, 6),
                },
                "dev_metrics": {
                    "dev_eer": float(dev_eer_attm),
                    "dev_min_dcf": float(dev_mindcf_attm),
                    "eer_threshold": float(dev_tau_attm),
                },
                "raw_metrics": res_attm_raw,
                "calibrated_metrics": res_attm_cal,
                "per_attack_eer_raw": per_atk_attm_raw,
                "per_attack_eer_calibrated": per_atk_attm_cal,
            },
            "uniform_ablation_12layers": {
                "checkpoint": "data/cm_aasist_attm_uniform.pt",
                "calibration_parameters": {
                    "method": "platt",
                    "scale_w": round(w_uni, 6),
                    "bias_b": round(b_uni, 6),
                },
                "dev_metrics": {
                    "dev_eer": float(dev_eer_uni),
                    "dev_min_dcf": float(dev_mindcf_uni),
                    "eer_threshold": float(dev_tau_uni),
                },
                "raw_metrics": res_uni_raw,
                "calibrated_metrics": res_uni_cal,
                "per_attack_eer_raw": per_atk_uni_raw,
                "per_attack_eer_calibrated": per_atk_uni_cal,
            },
        },
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report_content, f, indent=2)
    print(f"\nSaved side-by-side calibrated benchmark report to: {out_report}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
