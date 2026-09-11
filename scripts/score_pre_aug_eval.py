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

manifest_path = Path(r'A:\cm-corpus\manifests\eval_available.tsv')
sample_limit = 20000

records = []
with open(manifest_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    for row in reader:
        records.append((row[0], 1 if row[1] == 'bonafide' else 0, row[2]))
        if len(records) >= sample_limit:
            break

device = 'cuda'
ckpt = torch.load('data/pre_aug_cm_aasist.pt', map_location='cpu', weights_only=False)
model = AASISTLHead(in_dim=768).to(device)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').half().to(device)
wavlm.eval()

def load_audio(item):
    p, label, atk = item
    try:
        x, sr = sf.read(p, dtype='float32')
        w = prepare_audio_window(x, sr=sr, target_duration=4.0, random_crop=False)
        return w, label, atk
    except Exception as e:
        return np.zeros(64000, dtype=np.float32), label, atk

batch_size = 64
bon_scores = []
spf_scores = []
atk_scores = {}

t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as executor:
    for i in range(0, len(records), batch_size):
        batch_items = records[i:i+batch_size]
        processed = list(executor.map(load_audio, batch_items))
        audios = np.stack([p[0] for p in processed])
        labels = [p[1] for p in processed]
        atks = [p[2] for p in processed]
        audio_t = torch.from_numpy(audios).half().to(device)
        with torch.no_grad():
            hidden = wavlm(audio_t, output_hidden_states=True).hidden_states[-1].float()
            _, logits = model(hidden)
            raw_llrs = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        for s, y, atk in zip(raw_llrs, labels, atks):
            score = float(s)
            if y == 1:
                bon_scores.append(score)
            else:
                spf_scores.append(score)
                if atk not in atk_scores:
                    atk_scores[atk] = []
                atk_scores[atk].append(score)
        if (i // batch_size) % 50 == 0 and i > 0:
            print(f"Scored {i}/{len(records)} ({i/(time.time()-t0):.1f} utts/s)")

bon = np.array(bon_scores)
spf = np.array(spf_scores)
res = evaluate_system(bon, spf)
print('=== Pre-augmentation checkpoint on Eval 20k slice ===')
print('minDCF    :', round(res['min_dcf'], 4))
print('actDCF    :', round(res['act_dcf'], 4))
print('actDCF(B) :', round(res['act_dcf_bayes'], 4))
print('CLLR      :', round(res['cllr'], 4))
print('Pooled EER:', round(res['eer'] * 100.0, 3), '%')

per_atk = {}
for atk, scores in sorted(atk_scores.items()):
    s_arr = np.array(scores)
    if len(s_arr) > 0 and len(bon) > 0:
        atk_eer, _ = compute_eer(bon, s_arr)
        per_atk[atk] = round(atk_eer * 100.0, 3)
        print(f"  Attack {atk:<6}: {per_atk[atk]}%")

with open('reports/cm_aasist_eval_partition_pre_aug.json', 'w') as f:
    json.dump({'checkpoint': 'data/pre_aug_cm_aasist.pt', 'metrics': res, 'per_attack_eer': per_atk}, f, indent=2)
print("Saved reports/cm_aasist_eval_partition_pre_aug.json")
