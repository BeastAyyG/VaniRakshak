import csv, json, time, sys
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

slice_manifest = Path('data/eval_benchmark/eval_slice_20k.tsv')
records = []
with open(slice_manifest, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for r in reader:
        records.append((r[0], 1 if r[1] == 'bonafide' else 0, r[2]))

print(f'Evaluating {len(records)} records')

device = 'cuda'
wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').half().to(device)
wavlm.eval()

# Load models
ckpt_pre = torch.load('data/pre_aug_cm_aasist.pt', map_location='cpu')
m_pre = AASISTLHead(in_dim=768).to(device)
m_pre.load_state_dict(ckpt_pre['state_dict'], strict=False)
m_pre.eval()

ckpt_post = torch.load('data/cm_aasist.pt', map_location='cpu')
m_post = AASISTLHead(in_dim=768).to(device)
m_post.load_state_dict(ckpt_post['state_dict'], strict=False)
m_post.eval()

def load_audio(item):
    p, label, atk = item
    try:
        x, sr = sf.read(p, dtype='float32')
        w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
        return w, label, atk
    except Exception as e:
        return np.zeros(64000, dtype=np.float32), label, atk

batch_size = 64
bon_pre, spf_pre, atk_pre = [], [], {}
bon_post, spf_post, atk_post = [], [], {}

t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        proc = list(ex.map(load_audio, batch))
        audios = np.stack([p[0] for p in proc])
        labels = [p[1] for p in proc]
        atks = [p[2] for p in proc]
        
        at = torch.from_numpy(audios).half().to(device)
        with torch.no_grad():
            feat = wavlm(at, output_hidden_states=True).hidden_states[-1].float()
            _, l_pre = m_pre(feat)
            _, l_post = m_post(feat)
            llr_pre = (l_pre[:, 1] - l_pre[:, 0]).cpu().numpy()
            llr_post = (l_post[:, 1] - l_post[:, 0]).cpu().numpy()
            
        for lp, lpo, y, atk in zip(llr_pre, llr_post, labels, atks):
            sp, spo = float(lp), float(lpo)
            if y == 1:
                bon_pre.append(sp)
                bon_post.append(spo)
            else:
                spf_pre.append(sp)
                spf_post.append(spo)
                if atk not in atk_pre: atk_pre[atk] = []
                if atk not in atk_post: atk_post[atk] = []
                atk_pre[atk].append(sp)
                atk_post[atk].append(spo)
        if (i // batch_size) % 50 == 0 and i > 0:
            print(f'Scored {i}/{len(records)} ({i/(time.time()-t0):.1f} utts/s)')

bon_pre_a = np.array(bon_pre)
spf_pre_a = np.array(spf_pre)
bon_post_a = np.array(bon_post)
spf_post_a = np.array(spf_post)

res_pre = evaluate_system(bon_pre_a, spf_pre_a)
res_post = evaluate_system(bon_post_a, spf_post_a)

print('=== UNCALIBRATED (RAW LLR) COMPARISON ===')
print('Pre-Aug  EER:', round(res_pre['eer']*100, 3), 'minDCF:', round(res_pre['min_dcf'], 4), 'actDCF:', round(res_pre['act_dcf'], 4), 'actDCF_bayes:', round(res_pre['act_dcf_bayes'], 4), 'CLLR:', round(res_pre['cllr'], 4))
print('Post-Aug EER:', round(res_post['eer']*100, 3), 'minDCF:', round(res_post['min_dcf'], 4), 'actDCF:', round(res_post['act_dcf'], 4), 'actDCF_bayes:', round(res_post['act_dcf_bayes'], 4), 'CLLR:', round(res_post['cllr'], 4))

w = 0.369718
b = -3.616373
res_pre_cal = evaluate_system(w * bon_pre_a + b, w * spf_pre_a + b)
res_post_cal = evaluate_system(w * bon_post_a + b, w * spf_post_a + b)

print('=== CALIBRATED (PLATT SCALED) COMPARISON ===')
print('Pre-Aug (cal)  EER:', round(res_pre_cal['eer']*100, 3), 'minDCF:', round(res_pre_cal['min_dcf'], 4), 'actDCF:', round(res_pre_cal['act_dcf'], 4), 'actDCF_bayes:', round(res_pre_cal['act_dcf_bayes'], 4), 'CLLR:', round(res_pre_cal['cllr'], 4))
print('Post-Aug (cal) EER:', round(res_post_cal['eer']*100, 3), 'minDCF:', round(res_post_cal['min_dcf'], 4), 'actDCF:', round(res_post_cal['act_dcf'], 4), 'actDCF_bayes:', round(res_post_cal['act_dcf_bayes'], 4), 'CLLR:', round(res_post_cal['cllr'], 4))

def get_per_attack(bon, atk_dict, w_scale=1.0, b_bias=0.0):
    res = {}
    for atk, sc in sorted(atk_dict.items()):
        sc_a = np.array(sc) * w_scale + b_bias
        e, _ = compute_eer(bon * w_scale + b_bias, sc_a)
        res[atk] = round(e * 100.0, 3)
    return res

per_atk_pre = get_per_attack(bon_pre_a, atk_pre)
per_atk_post = get_per_attack(bon_post_a, atk_post)

rep_pre_raw = {
    'checkpoint': 'data/pre_aug_cm_aasist.pt',
    'partition': 'eval_slice_20k',
    'n_bonafide': len(bon_pre),
    'n_spoof': len(spf_pre),
    'metrics': res_pre,
    'per_attack_eer': per_atk_pre
}
rep_post_raw = {
    'checkpoint': 'data/cm_aasist.pt',
    'partition': 'eval_slice_20k',
    'n_bonafide': len(bon_post),
    'n_spoof': len(spf_post),
    'metrics': res_post,
    'per_attack_eer': per_atk_post
}
with open('reports/cm_aasist_eval_partition_pre_aug.json', 'w') as f:
    json.dump(rep_pre_raw, f, indent=2)

with open('reports/cm_aasist_eval_partition.json', 'w') as f:
    json.dump(rep_post_raw, f, indent=2)

print('Updated reports/cm_aasist_eval_partition_pre_aug.json and reports/cm_aasist_eval_partition.json (raw on raw)!')
