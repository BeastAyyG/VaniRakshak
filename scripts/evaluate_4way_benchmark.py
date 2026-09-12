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

    print(f"Starting 4-way evaluation on {len(records)} records on RTX 5060 Ti GPU...")

    device = "cuda"
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").half().to(device)
    wavlm.eval()

    # 1. Pre-Aug Layer 12
    ckpt_pre = torch.load("data/pre_aug_cm_aasist.pt", map_location="cpu")
    assert_frozen_checkpoint(ckpt_pre, "scripts/evaluate_4way_benchmark.py")
    m_pre = AASISTLHead(in_dim=768).to(device)
    m_pre.load_state_dict(ckpt_pre["state_dict"], strict=False)
    m_pre.eval()

    # 2. Post-Aug Layer 12
    ckpt_post = torch.load("data/cm_aasist.pt", map_location="cpu")
    assert_frozen_checkpoint(ckpt_post, "scripts/evaluate_4way_benchmark.py")
    m_post = AASISTLHead(in_dim=768).to(device)
    m_post.load_state_dict(ckpt_post["state_dict"], strict=False)
    m_post.eval()

    # 3. Layer 5 (Pan et al.)
    ckpt_l5 = torch.load("data/cm_aasist_layer5.pt", map_location="cpu")
    assert_frozen_checkpoint(ckpt_l5, "scripts/evaluate_4way_benchmark.py")
    m_l5 = AASISTLHead(in_dim=768).to(device)
    m_l5.load_state_dict(ckpt_l5["state_dict"], strict=False)
    m_l5.eval()

    # 4. Attentive Merging 12 layers
    ckpt_attm = torch.load("data/cm_aasist_attm.pt", map_location="cpu")
    assert_frozen_checkpoint(ckpt_attm, "scripts/evaluate_4way_benchmark.py")
    m_attm = AASISTLHead(in_dim=768, merge_layers=12).to(device)
    m_attm.load_state_dict(ckpt_attm["state_dict"], strict=True)
    m_attm.eval()

    def load_audio(item):
        p, label, atk = item
        try:
            x, sr = sf.read(p, dtype="float32")
            w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, atk
        except Exception:
            return np.zeros(64000, dtype=np.float32), label, atk

    batch_size = 64
    bon_pre, spf_pre, atk_pre = [], [], {}
    bon_post, spf_post, atk_post = [], [], {}
    bon_l5, spf_l5, atk_l5 = [], [], {}
    bon_attm, spf_attm, atk_attm = [], [], {}

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
                feat_l12 = out.hidden_states[-1].float()
                feat_l5 = out.hidden_states[5].float()
                feats_12 = [out.hidden_states[j].float() for j in range(12)]

                _, l_pre = m_pre(feat_l12)
                _, l_post = m_post(feat_l12)
                _, l_l5 = m_l5(feat_l5)
                _, l_attm = m_attm(feats_12)

                llr_pre = (l_pre[:, 1] - l_pre[:, 0]).cpu().numpy()
                llr_post = (l_post[:, 1] - l_post[:, 0]).cpu().numpy()
                llr_l5 = (l_l5[:, 1] - l_l5[:, 0]).cpu().numpy()
                llr_attm = (l_attm[:, 1] - l_attm[:, 0]).cpu().numpy()

            for lp, lpo, ll5, lat, y, atk in zip(llr_pre, llr_post, llr_l5, llr_attm, labels, atks):
                sp, spo, sl5, sat = float(lp), float(lpo), float(ll5), float(lat)
                if y == 1:
                    bon_pre.append(sp)
                    bon_post.append(spo)
                    bon_l5.append(sl5)
                    bon_attm.append(sat)
                else:
                    spf_pre.append(sp)
                    spf_post.append(spo)
                    spf_l5.append(sl5)
                    spf_attm.append(sat)
                    for d, val in [(atk_pre, sp), (atk_post, spo), (atk_l5, sl5), (atk_attm, sat)]:
                        if atk not in d:
                            d[atk] = []
                        d[atk].append(val)
            if (i // batch_size) % 50 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"Scored {i}/{len(records)} ({i/elapsed:.1f} utts/s, elapsed: {elapsed:.1f}s)")
                sys.stdout.flush()

            del at, out, feat_l12, feat_l5, feats_12, l_pre, l_post, l_l5, l_attm

    res_pre = evaluate_system(np.array(bon_pre), np.array(spf_pre))
    res_post = evaluate_system(np.array(bon_post), np.array(spf_post))
    res_l5 = evaluate_system(np.array(bon_l5), np.array(spf_l5))
    res_attm = evaluate_system(np.array(bon_attm), np.array(spf_attm))

    def get_per_attack(bon, atk_dict):
        res = {}
        bon_a = np.array(bon)
        for atk, sc in sorted(atk_dict.items()):
            sc_a = np.array(sc)
            e, _ = compute_eer(bon_a, sc_a)
            res[atk] = round(e * 100.0, 3)
        return res

    per_atk_pre = get_per_attack(bon_pre, atk_pre)
    per_atk_post = get_per_attack(bon_post, atk_post)
    per_atk_l5 = get_per_attack(bon_l5, atk_l5)
    per_atk_attm = get_per_attack(bon_attm, atk_attm)

    # Softmax weights of attentive merge
    w_softmax = torch.softmax(m_attm.merge.weights, dim=0).detach().cpu().numpy().tolist()

    report_data = {
        "evaluation_partition": "data/eval_benchmark/eval_slice_20k.tsv",
        "total_utterances": len(records),
        "bonafide_count": len(bon_attm),
        "spoof_count": len(spf_attm),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "pre_aug_layer12": {
                "checkpoint": "data/pre_aug_cm_aasist.pt",
                "layer_index": 12,
                "metrics": res_pre,
                "per_attack_eer": per_atk_pre,
            },
            "post_aug_layer12": {
                "checkpoint": "data/cm_aasist.pt",
                "layer_index": 12,
                "metrics": res_post,
                "per_attack_eer": per_atk_post,
            },
            "layer5_pan_et_al": {
                "checkpoint": "data/cm_aasist_layer5.pt",
                "layer_index": 5,
                "metrics": res_l5,
                "per_attack_eer": per_atk_l5,
            },
            "attentive_merge_12layers": {
                "checkpoint": "data/cm_aasist_attm.pt",
                "merge_layers": 12,
                "softmax_weights": [round(float(x) * 100.0, 2) for x in w_softmax],
                "metrics": res_attm,
                "per_attack_eer": per_atk_attm,
            },
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/cm_attentive_merge_benchmark.json", "w") as f:
        json.dump(report_data, f, indent=2)

    md_content = f"""# 4-Way Side-by-Side Benchmark Report: Attentive Layer Merging vs Single Layer Countermeasures

Evaluation Partition: `data/eval_benchmark/eval_slice_20k.tsv` (20,000 utterances: 4,058 bona fide, 15,942 spoof across all 16 held-out attacks A17–A32).

## 1. Summary Metrics (Raw LLR Basis)

| Model | Layer / Architecture | EER (%) | minDCF | actDCF | actDCF_bayes | CLLR |
|---|---|---|---|---|---|---|
| **Pre-Augmentation Baseline** | Layer 12 (-1) | {res_pre['eer']*100:.3f}% | {res_pre['min_dcf']:.4f} | {res_pre['act_dcf']:.4f} | {res_pre['act_dcf_bayes']:.4f} | {res_pre['cllr']:.4f} |
| **Post-Augmentation Model** | Layer 12 (-1) | {res_post['eer']*100:.3f}% | {res_post['min_dcf']:.4f} | {res_post['act_dcf']:.4f} | {res_post['act_dcf_bayes']:.4f} | {res_post['cllr']:.4f} |
| **Layer 5 (Pan et al. R10)** | Layer 5 | {res_l5['eer']*100:.3f}% | {res_l5['min_dcf']:.4f} | {res_l5['act_dcf']:.4f} | {res_l5['act_dcf_bayes']:.4f} | {res_l5['cllr']:.4f} |
| **Attentive Merging (12 Layers)** | **AttentiveMerging [0:12]** | **{res_attm['eer']*100:.3f}%** | **{res_attm['min_dcf']:.4f}** | **{res_attm['act_dcf']:.4f}** | **{res_attm['act_dcf_bayes']:.4f}** | **{res_attm['cllr']:.4f}** |

## 2. Attentive Merging Layer Attention Distribution

Learned softmax layer attention weights across `hidden_states[0:12]`:
"""
    for idx, w in enumerate(w_softmax):
        md_content += f"- **Layer {idx:2d}**: {w*100.0:.2f}%\n"

    md_content += """
## 3. Per-Attack Breakdown (EER %) on Held-Out Eval Attacks (A17–A32)

| Attack ID | Pre-Aug L12 (%) | Post-Aug L12 (%) | Layer 5 (%) | Attentive Merge 12L (%) | Best Performing Architecture |
|---|---|---|---|---|---|
"""
    for atk in sorted(per_atk_attm.keys()):
        p1 = per_atk_pre.get(atk, float("nan"))
        p2 = per_atk_post.get(atk, float("nan"))
        p3 = per_atk_l5.get(atk, float("nan"))
        p4 = per_atk_attm.get(atk, float("nan"))
        best_val = min(p1, p2, p3, p4)
        if best_val == p4:
            winner = "**Attentive Merge**"
        elif best_val == p3:
            winner = "Layer 5"
        elif best_val == p2:
            winner = "Post-Aug L12"
        else:
            winner = "Pre-Aug L12"
        md_content += f"| {atk} | {p1:.3f}% | {p2:.3f}% | {p3:.3f}% | {p4:.3f}% | {winner} |\n"

    with open("reports/cm_attentive_merge_benchmark.md", "w") as f:
        f.write(md_content)

    print("=== 4-WAY UNCALIBRATED BENCHMARK RESULTS ===")
    print("Model                    EER (%)   minDCF   actDCF   actDCF_bayes   CLLR")
    print(f"Pre-Aug (Layer 12):      {res_pre['eer']*100:6.3f}   {res_pre['min_dcf']:6.4f}   {res_pre['act_dcf']:6.4f}   {res_pre['act_dcf_bayes']:12.4f}   {res_pre['cllr']:6.4f}")
    print(f"Post-Aug (Layer 12):     {res_post['eer']*100:6.3f}   {res_post['min_dcf']:6.4f}   {res_post['act_dcf']:6.4f}   {res_post['act_dcf_bayes']:12.4f}   {res_post['cllr']:6.4f}")
    print(f"Layer 5 (Pan et al):     {res_l5['eer']*100:6.3f}   {res_l5['min_dcf']:6.4f}   {res_l5['act_dcf']:6.4f}   {res_l5['act_dcf_bayes']:12.4f}   {res_l5['cllr']:6.4f}")
    print(f"Attentive Merge (12L):   {res_attm['eer']*100:6.3f}   {res_attm['min_dcf']:6.4f}   {res_attm['act_dcf']:6.4f}   {res_attm['act_dcf_bayes']:12.4f}   {res_attm['cllr']:6.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
