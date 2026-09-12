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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cm_frontend import assert_frozen_checkpoint
from benchmarks.compute_mindcf import compute_eer, compute_mindcf, evaluate_system
from ml.aasist_model import AASISTLHead
from ml.augment import prepare_audio_window

def main():
    slice_manifest = Path("data/eval_benchmark/eval_slice_20k.tsv")
    records = []
    with open(slice_manifest, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for r in reader:
            records.append((r[0], 1 if r[1] == "bonafide" else 0, r[2]))

    print(f"Scoring uniform ablation checkpoint on {len(records)} records on GPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    if device == "cuda":
        wavlm = wavlm.half().to(device)
    wavlm.eval()

    ckpt_path = "data/cm_aasist_attm_uniform.pt"
    if not os.path.exists(ckpt_path):
        print(f"{ckpt_path} not found; generating from data/cm_aasist_attm.pt...")
        ck = torch.load("data/cm_aasist_attm.pt", map_location="cpu")
        assert_frozen_checkpoint(ck, "scripts/evaluate_uniform_ablation.py")
        sd = ck["state_dict"]
        k = [x for x in sd if x.endswith("merge.weights")][0]
        print("Learned softmax before zeroing:", torch.softmax(sd[k], dim=0).tolist())
        sd[k] = torch.zeros_like(sd[k])
        ck["uniformized_from"] = "data/cm_aasist_attm.pt"
        torch.save(ck, ckpt_path)
        print(f"Saved {ckpt_path}")

    ckpt_uni = torch.load(ckpt_path, map_location="cpu")

    assert_frozen_checkpoint(ckpt_uni, "scripts/evaluate_uniform_ablation.py")
    m_uni = AASISTLHead(in_dim=768, merge_layers=12).to(device)
    m_uni.load_state_dict(ckpt_uni["state_dict"], strict=True)
    m_uni.eval()

    def load_audio(item):
        p, label, atk = item
        try:
            x, sr = sf.read(p, dtype="float32")
            w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, atk
        except Exception:
            return np.zeros(64000, dtype=np.float32), label, atk

    batch_size = 64
    bon_uni, spf_uni, atk_uni = [], [], {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            proc = list(ex.map(load_audio, batch))
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
                _, l_uni = m_uni(feats_12)
                llr_uni = (l_uni[:, 1] - l_uni[:, 0]).cpu().numpy()

            for lu, y, atk in zip(llr_uni, labels, atks):
                su = float(lu)
                if y == 1:
                    bon_uni.append(su)
                else:
                    spf_uni.append(su)
                    if atk not in atk_uni:
                        atk_uni[atk] = []
                    atk_uni[atk].append(su)

            if (i // batch_size) % 50 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"Scored {i}/{len(records)} ({i/elapsed:.1f} utts/s, elapsed: {elapsed:.1f}s)")
                sys.stdout.flush()

            del at, out, feats_12, l_uni

    res_uni = evaluate_system(np.array(bon_uni), np.array(spf_uni))

    def get_per_attack(bon, atk_dict):
        res = {}
        bon_a = np.array(bon)
        for atk, sc in sorted(atk_dict.items()):
            sc_a = np.array(sc)
            e, _ = compute_eer(bon_a, sc_a)
            res[atk] = round(e * 100.0, 3)
        return res

    per_atk_uni = get_per_attack(bon_uni, atk_uni)

    with open("reports/cm_attentive_merge_benchmark.json", "r") as f:
        existing_benchmark = json.load(f)

    models_dict = existing_benchmark["models"]

    models_dict["uniform_ablation_12layers"] = {
        "checkpoint": "data/cm_aasist_attm_uniform.pt",
        "merge_layers": 12,
        "uniform_weights": [100.0 / 12] * 12,
        "metrics": res_uni,
        "per_attack_eer": per_atk_uni,
    }

    report_data = {
        "evaluation_partition": "data/eval_benchmark/eval_slice_20k.tsv",
        "total_utterances": len(records),
        "bonafide_count": len(bon_uni),
        "spoof_count": len(spf_uni),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models_dict,
    }

    os.makedirs("reports", exist_ok=True)
    out_json = "reports/cm_uniform_ablation_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved uniform ablation report to {out_json}")

    print("\n=== UNIFORM ABLATION RESULTS ===")
    print(f"EER (%):      {res_uni['eer']*100:.3f}%")
    print(f"minDCF:       {res_uni['min_dcf']:.4f}")
    print(f"actDCF:       {res_uni['act_dcf']:.4f}")
    print(f"actDCF_bayes: {res_uni['act_dcf_bayes']:.4f}")
    print(f"CLLR:         {res_uni['cllr']:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
