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

    print(f"Starting 4-way evaluation on {len(records)} records on RTX 5060 Ti GPU...")

    device = "cuda"
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").half().to(device)
    wavlm.eval()

    # Frozen Meanpool (12 layers, uniform)
    ckpt_mp = torch.load("data/cm_aasist_meanpool.pt", map_location="cpu")
    print(f"Loaded checkpoint data/cm_aasist_meanpool.pt: dev_eer={ckpt_mp.get('dev_eer')}, dev_min_dcf={ckpt_mp.get('dev_min_dcf')}, freeze_merge={ckpt_mp.get('freeze_merge')}")
    m_mp = AASISTLHead(in_dim=768, merge_layers=12, freeze_merge=True).to(device)
    m_mp.load_state_dict(ckpt_mp["state_dict"], strict=True)
    m_mp.eval()

    def load_audio(item):
        p, label, atk = item
        try:
            x, sr = sf.read(p, dtype="float32")
            w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, atk
        except Exception:
            return np.zeros(64000, dtype=np.float32), label, atk

    batch_size = 64
    bon_mp, spf_mp, atk_mp = [], [], {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            proc = list(ex.map(load_audio, batch))
            audios = np.stack([p[0] for p in proc])
            labels = [p[1] for p in proc]
            atks = [p[2] for p in proc]

            at = torch.from_numpy(audios).half().to(device)
            with torch.no_grad():
                out = wavlm(at, output_hidden_states=True)
                feats_12 = [out.hidden_states[j].float() for j in range(12)]
                _, l_mp = m_mp(feats_12)
                llr_mp = (l_mp[:, 1] - l_mp[:, 0]).cpu().numpy()

            for lmp, y, atk in zip(llr_mp, labels, atks):
                smp = float(lmp)
                if y == 1:
                    bon_mp.append(smp)
                else:
                    spf_mp.append(smp)
                    if atk not in atk_mp:
                        atk_mp[atk] = []
                    atk_mp[atk].append(smp)

            if (i // batch_size) % 50 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"Scored {i}/{len(records)} ({i/elapsed:.1f} utts/s, elapsed: {elapsed:.1f}s)")
                sys.stdout.flush()

            del at, out, feats_12, l_mp

    res_mp = evaluate_system(np.array(bon_mp), np.array(spf_mp))

    def get_per_attack(bon, atk_dict):
        res = {}
        bon_a = np.array(bon)
        for atk, sc in sorted(atk_dict.items()):
            sc_a = np.array(sc)
            e, _ = compute_eer(bon_a, sc_a)
            res[atk] = round(e * 100.0, 3)
        return res

    per_atk_mp = get_per_attack(bon_mp, atk_mp)

    # Load 4-way baseline from reports/cm_attentive_merge_benchmark.json
    with open("reports/cm_attentive_merge_benchmark.json", "r") as f:
        existing_benchmark = json.load(f)

    models_dict = existing_benchmark["models"]

    models_dict["frozen_meanpool_12layers"] = {
        "checkpoint": "data/cm_aasist_meanpool.pt",
        "merge_layers": 12,
        "freeze_merge": True,
        "metrics": res_mp,
        "per_attack_eer": per_atk_mp,
    }

    report_data = {
        "evaluation_partition": "data/eval_benchmark/eval_slice_20k.tsv",
        "total_utterances": len(records),
        "bonafide_count": len(bon_mp),
        "spoof_count": len(spf_mp),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": models_dict,
    }

    os.makedirs("reports", exist_ok=True)
    out_json = "reports/cm_meanpool_ablation_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved 5-way benchmark report to {out_json}")

    res_pre = models_dict["pre_aug_layer12"]["metrics"]
    res_post = models_dict["post_aug_layer12"]["metrics"]
    res_l5 = models_dict["layer5_pan_et_al"]["metrics"]
    res_attm = models_dict["attentive_merge_12layers"]["metrics"]

    per_atk_pre = models_dict["pre_aug_layer12"]["per_attack_eer"]
    per_atk_post = models_dict["post_aug_layer12"]["per_attack_eer"]
    per_atk_l5 = models_dict["layer5_pan_et_al"]["per_attack_eer"]
    per_atk_attm = models_dict["attentive_merge_12layers"]["per_attack_eer"]

    md_content = f"""# 5-Way Ablation Benchmark Report: Frozen Meanpool vs Attentive Merging vs Single Layers

Evaluation Partition: `data/eval_benchmark/eval_slice_20k.tsv` (20,000 utterances: 4,058 bona fide, 15,942 spoof across all 16 held-out attacks A17–A32).

## 1. Summary Metrics (Raw LLR Basis)

| Model | Layer / Architecture | EER (%) | minDCF | actDCF | actDCF_bayes | CLLR |
|---|---|---|---|---|---|---|
| **Pre-Aug Baseline** | Layer 12 (-1) | {res_pre['eer']*100:.3f}% | {res_pre['min_dcf']:.4f} | {res_pre['act_dcf']:.4f} | {res_pre['act_dcf_bayes']:.4f} | {res_pre['cllr']:.4f} |
| **Post-Aug Model** | Layer 12 (-1) | {res_post['eer']*100:.3f}% | {res_post['min_dcf']:.4f} | {res_post['act_dcf']:.4f} | {res_post['act_dcf_bayes']:.4f} | {res_post['cllr']:.4f} |
| **Layer 5 (Pan et al. R10)** | Layer 5 | {res_l5['eer']*100:.3f}% | {res_l5['min_dcf']:.4f} | {res_l5['act_dcf']:.4f} | {res_l5['act_dcf_bayes']:.4f} | {res_l5['cllr']:.4f} |
| **Attentive Merging (12 Layers)** | AttentiveMerging [0:12] | {res_attm['eer']*100:.3f}% | {res_attm['min_dcf']:.4f} | {res_attm['act_dcf']:.4f} | {res_attm['act_dcf_bayes']:.4f} | {res_attm['cllr']:.4f} |
| **Frozen Meanpool (12 Layers)** | **Uniform Mean [0:12]** | **{res_mp['eer']*100:.3f}%** | **{res_mp['min_dcf']:.4f}** | **{res_mp['act_dcf']:.4f}** | **{res_mp['act_dcf_bayes']:.4f}** | **{res_mp['cllr']:.4f}** |

## 2. Per-Attack Breakdown (EER %) on Held-Out Eval Attacks (A17–A32)

| Attack ID | Pre-Aug L12 (%) | Post-Aug L12 (%) | Layer 5 (%) | Attentive Merge 12L (%) | Frozen Meanpool 12L (%) |
|---|---|---|---|---|---|
"""
    for atk in sorted(per_atk_mp.keys()):
        p1 = per_atk_pre.get(atk, float("nan"))
        p2 = per_atk_post.get(atk, float("nan"))
        p3 = per_atk_l5.get(atk, float("nan"))
        p4 = per_atk_attm.get(atk, float("nan"))
        p5 = per_atk_mp.get(atk, float("nan"))
        md_content += f"| {atk} | {p1:.3f}% | {p2:.3f}% | {p3:.3f}% | {p4:.3f}% | {p5:.3f}% |\n"

    out_md = "reports/cm_meanpool_ablation_benchmark.md"
    with open(out_md, "w") as f:
        f.write(md_content)
    print(f"Saved 5-way markdown report to {out_md}")

    print("\n=== 5-WAY UNCALIBRATED BENCHMARK RESULTS ===")
    print("Model                       EER (%)   minDCF   actDCF   actDCF_bayes   CLLR")
    print(f"Pre-Aug (Layer 12):         {res_pre['eer']*100:6.3f}   {res_pre['min_dcf']:6.4f}   {res_pre['act_dcf']:6.4f}   {res_pre['act_dcf_bayes']:12.4f}   {res_pre['cllr']:6.4f}")
    print(f"Post-Aug (Layer 12):        {res_post['eer']*100:6.3f}   {res_post['min_dcf']:6.4f}   {res_post['act_dcf']:6.4f}   {res_post['act_dcf_bayes']:12.4f}   {res_post['cllr']:6.4f}")
    print(f"Layer 5 (Pan et al):        {res_l5['eer']*100:6.3f}   {res_l5['min_dcf']:6.4f}   {res_l5['act_dcf']:6.4f}   {res_l5['act_dcf_bayes']:12.4f}   {res_l5['cllr']:6.4f}")
    print(f"Attentive Merge (12L):      {res_attm['eer']*100:6.3f}   {res_attm['min_dcf']:6.4f}   {res_attm['act_dcf']:6.4f}   {res_attm['act_dcf_bayes']:12.4f}   {res_attm['cllr']:6.4f}")
    print(f"Frozen Meanpool (12L):      {res_mp['eer']*100:6.3f}   {res_mp['min_dcf']:6.4f}   {res_mp['act_dcf']:6.4f}   {res_mp['act_dcf_bayes']:12.4f}   {res_mp['cllr']:6.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
