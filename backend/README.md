# VaniRakshak Enterprise Deep Learning Backend

> [!WARNING]
> **DEPRECATED — this tree is not wired into the product. Do not run it.**
>
> The serving backend is **`vanirakshak-backend/`**. It is the one started by
> `docker-compose.yml`, `make backend`, `npm run dev:all` and
> `scripts/integration_check.py`, and it is the only backend the frontend
> (`frontend/`) talks to.
>
> This directory is an earlier, parallel implementation that was never
> connected to the serving path. It duplicates risk-fusion logic under a
> different vocabulary (`backend/app/engine/risk_fusion.py` emits
> `AUTHENTIC_TARGET` / `SYNTHETIC_VOICE_CLONE` / … from a different model than
> `vanirakshak-backend/vanirakshak/risk/fusion.py`), so its numbers are **not
> comparable** to the product's and must not be quoted as the system's results.
> It also requires `torchaudio` and `librosa`, which the product does not.
>
> It is kept only because `audit_detectors.py` and `audit_heavy_backend.py`
> deliberately score its 429-line DSP as a comparison baseline. Nothing else
> imports it. See `AUDIT_REPORT.md` §3/§6 and §10.
>
> **New work goes in `vanirakshak-backend/`.**

Enterprise FastAPI backend leveraging deep neural acoustic architectures:
- Intermediate 768-dimensional representations from `microsoft/wavlm-base-plus` across 2.0s sliding windows.
- Speaker verification using 192-dimensional embeddings from `speechbrain/spkrec-ecapa-voxceleb`.
- High-frequency Wiener entropy (spectral flatness) and vocoder phase boundary localization.
- Energy-based Voice Activity Detection (VAD).

## Architecture

```
backend/
├── Dockerfile           # Production container definition
├── requirements.txt     # Deep learning dependencies (torch, torchaudio, transformers, speechbrain)
├── app/
│   ├── main.py          # FastAPI application entrypoint & WebSocket /ws/stream handler
│   ├── config.py        # Hardware auto-detection, thresholds, model identifiers
│   ├── api/             # REST API routers (/api/v1/health, /api/v1/analyze/audio, /api/v1/enroll)
│   ├── core/            # In-memory audio tensor standardization and forensic signal DSP
│   ├── engine/          # Sliding window buffer and multi-factor risk fusion engine
│   └── models/          # WavLM feature extractor & ECAPA speaker verifier
└── tests/               # Forensic DSP and signal processing test suite
```

## Running the Backend

**Don't.** This tree is deprecated; see the warning above. To run the product:

```bash
make backend            # or: cd vanirakshak-backend && python run.py
```

(For the record, this legacy tree used to be started with
`uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload` — it is
reproduced here only so the historical command is not lost.)

## Lightweight Alternative

For resource-constrained laptop environments without CUDA GPUs or deep learning wheels, see `vanirakshak-backend/`, which provides a pure NumPy / SciPy implementation capable of sub-25ms latency on a single CPU core.
