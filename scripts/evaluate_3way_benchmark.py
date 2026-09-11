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

REPO_ROOT = Path('c:/VaniRakshak')
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.compute_mindcf import compute_eer, compute_mindcf, evaluate_system
from ml.aasist_model import AASISTLHead
from ml.augment import prepare_audio_window

def main():
    slice_manifest = Path('data/eval_benchmark/eval_slice_20k.tsv')
    records = []
    with open(slice_manifest, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for r in reader:
            records.append((r[0], 1 if r[1] == 'bonafide' else 0, r[2]))

    print(f'Starting 3-way evaluation on {len(records)} records on RTX 5060 Ti GPU...')

    device = 'cuda'
    wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').half().to(device)
    wavlm.eval()

    ckpt_pre = torch.load('data/pre_aug_cm_aasist.pt', map_location='cpu')
    m_pre = AASISTLHead(in_dim=768).to(device)
    m_pre.load_state_dict(ckpt_pre['state_dict'], strict=False)
    m_pre.eval()

    ckpt_post = torch.load('data/cm_aasist.pt', map_location='cpu')
    m_post = AASISTLHead(in_dim=768).to(device)
    m_post.load_state_dict(ckpt_post['state_dict'], strict=False)
    m_post.eval()

    ckpt_l5 = torch.load('data/cm_aasist_layer5.pt', map_location='cpu')
    m_l5 = AASISTLHead(in_dim=768).to(device)
    m_l5.load_state_dict(ckpt_l5['state_dict'], strict=False)
    m_l5.eval()

    def load_audio(item):
        p, label, atk = item
        try:
            x, sr = sf.read(p, dtype='float32')
            w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
            return w, label, atk
        except Exception:
            return np.zeros(64000, dtype=np.float32), label, atk

    batch_size = 64
    bon_pre, spf_pre, atk_pre = [], [], {}
    bon_post, spf_post, atk_post = [], [], {}
    bon_l5, spf_l5, atk_l5 = [], [], {}

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

                _, l_pre = m_pre(feat_l12)
                _, l_post = m_post(feat_l12)
                _, l_l5 = m_l5(feat_l5)

                llr_pre = (l_pre[:, 1] - l_pre[:, 0]).cpu().numpy()
                llr_post = (l_post[:, 1] - l_post[:, 0]).cpu().numpy()
                llr_l5 = (l_l5[:, 1] - l_l5[:, 0]).cpu().numpy()

            for lp, lpo, ll5, y, atk in zip(llr_pre, llr_post, llr_l5, labels, atks):
                sp, spo, sl5 = float(lp), float(lpo), float(ll5)
                if y == 1:
                    bon_pre.append(sp)
                    bon_post.append(spo)
                    bon_l5.append(sl5)
                else:
                    spf_pre.append(sp)
                    spf_post.append(spo)
                    spf_l5.append(sl5)
                    for d, val in [(atk_pre, sp), (atk_post, spo), (atk_l5, sl5)]:
                        if atk not in d:
                            d[atk] = []
                        d[atk].append(val)
            if (i // batch_size) % 50 == 0 and i > 0:
                elapsed = time.time() - t0
                print(f"Scored {i}/{len(records)} ({i/elapsed:.1f} utts/s, elapsed: {elapsed:.1f}s)")

    res_pre = evaluate_system(np.array(bon_pre), np.array(spf_pre))
    res_post = evaluate_system(np.array(bon_post), np.array(spf_post))
    res_l5 = evaluate_system(np.array(bon_l5), np.array(spf_l5))

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

    report_data = {
        "evaluation_partition": "data/eval_benchmark/eval_slice_20k.tsv",
        "total_utterances": len(records),
        "bonafide_count": len(bon_l5),
        "spoof_count": len(spf_l5),
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
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/cm_layer5_vs_layer12_benchmark.json", "w") as f:
        json.dump(report_data, f, indent=2)

    md_content = f"""# Side-by-Side Benchmark Report: WavLM Layer 5 vs Layer 12

Evaluation Partition: data/eval_benchmark/eval_slice_20k.tsv (20,000 utterances: 4,058 bona fide, 15,942 spoof across all 16 held-out attacks A17–A32).

## 1. Summary Metrics (Raw LLR Basis)

| Model | Layer | EER (%) | minDCF | actDCF | actDCF_bayes | CLLR |
|---|---|---|---|---|---|---|
| **Pre-Augmentation Baseline** | 12 (-1) | {res_pre['eer']*100:.3f}% | {res_pre['min_dcf']:.4f} | {res_pre['act_dcf']:.4f} | {res_pre['act_dcf_bayes']:.4f} | {res_pre['cllr']:.4f} |
| **Post-Augmentation Model** | 12 (-1) | {res_post['eer']*100:.3f}% | {res_post['min_dcf']:.4f} | {res_post['act_dcf']:.4f} | {res_post['act_dcf_bayes']:.4f} | {res_post['cllr']:.4f} |
| **Layer 5 (Pan et al. R10)** | **5** | **{res_l5['eer']*100:.3f}%** | **{res_l5['min_dcf']:.4f}** | **{res_l5['act_dcf']:.4f}** | **{res_l5['act_dcf_bayes']:.4f}** | **{res_l5['cllr']:.4f}** |

## 2. Per-Attack Breakdown (EER %) on Held-Out Eval Attacks (A17–A32)

| Attack ID | Pre-Aug Layer 12 (%) | Post-Aug Layer 12 (%) | Layer 5 (Pan et al.) (%) | Winner |
|---|---|---|---|---|
"""
    for atk in sorted(per_atk_l5.keys()):
        p1 = per_atk_pre.get(atk, float("nan"))
        p2 = per_atk_post.get(atk, float("nan"))
        p3 = per_atk_l5.get(atk, float("nan"))
        best = min(p1, p2, p3)
        winner = "Layer 5" if best == p3 else ("Post-Aug" if best == p2 else "Pre-Aug")
        md_content += f"| {atk} | {p1:.3f}% | {p2:.3f}% | {p3:.3f}% | {winner} |\n"

    with open("reports/cm_layer5_vs_layer12_benchmark.md", "w") as f:
        f.write(md_content)

    print("=== 3-WAY UNCALIBRATED BENCHMARK RESULTS ===")
    print("Model                 EER (%)   minDCF   actDCF   actDCF_bayes   CLLR")
    print(f"Pre-Aug (Layer 12):   {res_pre['eer']*100:6.3f}   {res_pre['min_dcf']:6.4f}   {res_pre['act_dcf']:6.4f}   {res_pre['act_dcf_bayes']:12.4f}   {res_pre['cllr']:6.4f}")
    print(f"Post-Aug (Layer 12):  {res_post['eer']*100:6.3f}   {res_post['min_dcf']:6.4f}   {res_post['act_dcf']:6.4f}   {res_post['act_dcf_bayes']:12.4f}   {res_post['cllr']:6.4f}")
    print(f"Layer 5 (Pan et al):  {res_l5['eer']*100:6.3f}   {res_l5['min_dcf']:6.4f}   {res_l5['act_dcf']:6.4f}   {res_l5['act_dcf_bayes']:12.4f}   {res_l5['cllr']:6.4f}")

if __name__ == '__main__':
    main()
