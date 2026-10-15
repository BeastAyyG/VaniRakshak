# SESSION HANDOFF — VaniRakshak (lab-beast)

The previous Claude Code session (e836fe74) hit its context limit at ~00:24 IST on 2026-09-29.
Training and GPU pipelines run DETACHED and are unaffected. This file is the full state
you need to continue. Read it, then check the live state files before acting.

## Machine and repo

- Host: lab-beast (TON-618), repo `C:\VaniRakshak`, branch `feat/gpu-group1-group2`, HEAD `8ce0efb`.
- Corpus: `A:\cm-corpus\asvspoof5\`. Pipeline state: `C:\vr-out\pipeline\` (state_g1.json, state_g2.json, state_spk.json).
- GPU dashboard: http://127.0.0.1:8787 (also on 100.89.248.108:8787).

## What is running right now (as of 2026-09-29 ~01:00 IST)

1. **docs/16 arm C seed 42 training** (`ml/train_cm.py --merge-layers 12 --layer-index 5 --epochs 10
   --batch-size 64 --lr 5e-4 --weight-decay 1e-4 --patience 5`, no init weights, from scratch).
   Epoch 1 done: dev minDCF 0.5011, EER 5.18%. Checkpoint: `data/cm_docs16_C_s42.pt`.
   Epoch 2 was at ~91% at 00:55 IST. ~4.5 hr per arm-seed; budget 40 GPU-hr total.
2. **g1_telephony_heads**: scoring all checkpoints through telephony channels. Was 79/84 at 00:55 IST.
3. **g1_speaker_grouped**: DONE 00:23. Outputs in `reports/t1_decision_layer_speaker_grouped_gate.json`
   and `reports/t1_speaker_grouped_vs_iid_gate.json` (untracked, need committing later).
4. **Pending**: g2_futility, g2_train_rest (C43/R43/C44/R44), g2_eval, g2_adjudicate.

## Key numbers you must know

- STOP-C is TRIGGERED: incumbent misses 67.4% of A28 through G.711 (63.6% AMR-NB).
- Speaker-grouped T1 re-measurement: grouped is much harder than iid.
  Genuine block 0.12 -> 0.15; worst-attack spoof block 0.9975 -> 0.81.
- Fixed decision-layer params (from the grouped gate): fusion intercept -1.6,
  w_cm 1.5, w_asv 2.0, w_prosody 0.7, w_channel 0.9, w_context 0.5, w_semantic 0.4;
  operating threshold 4.7066 (5th pct of dev LLR); VR_CLONE_CM_LLR_MAX -2.4568;
  VR_CLONE_MIN_WINDOWS 3; VR_CHALLENGE_CM_LLR_MAX 0.5757; VR_CHALLENGE_MIN_WINDOWS 2;
  VR_CHALLENGE_TRIGGER_CONSECUTIVE 3.
- docs/16 arms differ ONLY in `--real-codec-p` (C: 0, R: 0.5 over G.711mu/A-law/AMR-NB/AMR-WB,
  applied to RAW audio pre-windowing via ml/real_codecs.py). docs/20 arm B checkpoints are
  context-only, NOT the control. ml/augment.py codec branch stays broken for reproducibility.

## Pending decisions (in order)

1. **Futility check** (next): after C42 and R42 are trained AND scored, if
   worst-narrowband-miss(R42) > worst-narrowband-miss(C42) + 1pp, cancel the remaining
   runs and call NO_EFFECT early.
2. **docs/16 final verdict** (ADOPT / IMPROVES_NOT_ENOUGH / NO_EFFECT / HARMS) per the
   P16-1..P16-3 criteria in docs/16. ADOPT only makes the best R seed a CANDIDATE;
   it must then pass T1 re-calibration, P1-T, and full STOP-C on the 608k gate split.
3. **Which checkpoint to build on** once g1_telephony_heads finishes (per-checkpoint thresholds).
4. Owner-only: G1 step-up policy, IndicConformer access (G4), merge/publish of this branch
   (gh on lab-beast is unauthenticated; PR #16 was updated from elsewhere).
5. After adjudication: fix ml/augment.py, G5 W1.c fusion refit (not for small agents).

## Integrity rules (docs/16 section 6)

- Do NOT touch ml/train_cm.py, ml/real_codecs.py, ml/augment.py, or eval code while runs are in flight.
- Crashed runs re-run from scratch, never resumed.
- docs/16 was committed BEFORE training started; verdicts use the fixed criteria, no tuning after the fact.

## Where truth lives now

Pipeline state files and logs, not the dead session. Watch:
`C:\vr-out\pipeline\state_g2.json` (futility gate), the newest training log under
`C:\VaniRakshak` (epoch/step/loss), and `nvidia-smi`.
An external monitor checks every 10 minutes and reports stalls, errors, and completions.
