# Privacy Compliance & API Reference

> **2026-09-09 note:** §4 and §5 below duplicated the WebSocket/REST contract
> and environment-variable table independently of `docs/PROTOCOL.md` and
> `docs/API_SPEC.md` — and had drifted from both (a pre-merge health payload
> missing `hardware`/`models`/`anchor`, no `semantic`/`target_speaker`
> blocks, no `POST /v1/verify-speaker` or `GET /v1/enroll`, several env vars
> missing). Three independent descriptions of the same contract is how this
> kind of drift happens in the first place, so §4/§5 now point at the single
> current source instead of maintaining a third copy. §1-3 (the DPDP/privacy
> architecture) are this document's unique content and remain in full below.

## 1. Regulatory Framework: India DPDP Act 2023 Compliance

The **Digital Personal Data Protection (DPDP) Act, 2023** establishes strict requirements for the processing, retention, and protection of digital personal data within India. Voice recordings constitute **biometric personal data** under Indian privacy law.

### DPDP Act §8(7) Compliance: Zero Raw-Audio Retention

Section 8(7) of the DPDP Act mandates:
> *"A Data Fiduciary shall erase personal data, upon the Data Principal withdrawing her consent or as soon as it is reasonable to assume that the specified purpose for which personal data was collected is no longer being served, whichever is earlier."*

VaniRakshak achieves compliance through **privacy-by-architecture**:

```
                                 ┌─────────────────────────────────┐
                                 │   Continuous Audio Ingestion    │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ VOLATILE RAM SLIDING BUFFER (SlidingAudioBuffer)                                                │
│ • Fixed ring size: 80,640 samples (approx. 161 KB in memory)                                    │
│ • Window size: 4.04 s | Hop size: 1.0 s                                                         │
│ • In-place circular overwrite: Audio frames auto-expire in < 5.04 seconds                       │
│ • ZERO disk writes: No temporary .wav files, no swap-disk dumps, no persistent object storage   │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DEFENSIVE MEMORY WIPING                                                                         │
│ • Call Hangup / Session Completion: buffer.wipe() zero-fills buffer memory (_buf.fill(0))       │
│ • Interlock Triggered / Anomaly Detected: Immediate volatile buffer purge                       │
└────────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FEATURE-ONLY CRYPTOGRAPHIC AUDIT LOG (AuditLog)                                                 │
│ • Non-invertible scalar metrics only (cm_score, asv_consistency, snr_db)                        │
│ • NO raw audio waveforms, NO high-dimensional invertible embeddings                             │
│ • SHA-256 hash-chained receipts for tamper-evident compliance audit trail                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. In-Memory Buffering Architecture (`SlidingAudioBuffer`)

The in-memory audio ring buffer is implemented in [`vanirakshak-backend/vanirakshak/privacy/buffer.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/privacy/buffer.py):

### Key Operational Characteristics
1. **Bounded Ring Memory Allocation:**
   $$\text{Capacity} = \text{window\_samples} + \text{hop\_samples} = \operatorname{round}(4.04 \times 16000) + \operatorname{round}(1.0 \times 16000) = 80,640\text{ samples}$$
   For 16-bit signed integers, this consumes exactly **$161,280\text{ bytes}$ ($\approx 161.3\text{ KB}$)** of volatile RAM per active call session.
2. **Circular FIFO Overwrite:**
   Incoming PCM frames advance `_write_pos`. When `_write_pos` reaches the buffer boundary, it wraps around to index `0`. Older audio frames are overwritten in-place. At no point can audio older than $5.04\text{ seconds}$ exist in memory.
3. **Defensive Wipe (`wipe()`):**
   When a call terminates, or upon transaction interlock execution, `buffer.wipe()` executes:
   ```python
   self._buf.fill(0)
   self._write_pos = 0
   self._filled = 0
   self._samples_since_hop = 0
   ```
   This ensures zero residual audio data remains in unmanaged RAM.

---

## 3. SHA-256 Append-Only Hash-Chained Audit Trail

To satisfy banking compliance auditing (RBI Master Directions on Fraud Risk Management and CERT-In logging directives) while adhering to DPDP Act privacy rules, VaniRakshak implements a **feature-only cryptographic audit log** in [`vanirakshak-backend/vanirakshak/privacy/audit.py`](file:///d:/VaniRakshak/vanirakshak-backend/vanirakshak/privacy/audit.py).

### Non-Invertibility Guarantee
The audit log stores strictly non-invertible scalar metrics:
- Session metadata: `session_id`, `caller_id`, `claimed_identity`, UTC timestamp
- Forensic outputs: `risk_score`, `tier` (`ALLOW`, `CHALLENGE`, `BLOCK`)
- Scalar features: `cm_score`, `cm_genuine_prob`, `asv_consistency`, `channel_anomaly`, `prosody_unnatural`, `snr_db`
- **Zero raw audio and zero raw neural embeddings are logged.** It is mathematically impossible to reconstruct the speaker's voice from these scalar telemetry records.

### Cryptographic Chaining
Every event is cryptographically bound to its predecessor via SHA-256:

$$\text{receipt\_hash}_k = \operatorname{SHA-256}\Big(\operatorname{canonical\_json}(\text{payload}_k) \Big)$$

where:
$$\text{payload}_k = \Big\{ \text{session\_id}, \; \text{caller\_id}, \; \text{claimed\_identity}, \; \text{ts}, \; \text{risk}, \; \text{tier}, \; \text{features}, \; \text{prev} = \text{receipt\_hash}_{k-1} \Big\}$$

The canonical JSON formatting uses strict key sorting and minimal whitespace (`separators=(",", ":")`), guaranteeing deterministic hash generation across heterogeneous systems.

### Verification (`verify_chain()`)
The audit log includes an automated integrity verification method:
```python
is_valid = audit_log.verify_chain()
```
If an insider or attacker alters, re-orders, or deletes any audit log entry, the hash chain breaks from that point forward, rendering tampering immediately evident.

---

## 4. API Reference

See **[docs/PROTOCOL.md](PROTOCOL.md)** for the WebSocket contract
(handshake, verdict schema, `metrics.asv_evaluated`, the always-present
`semantic` block, the challenge object, close codes) and
**[docs/API_SPEC.md](API_SPEC.md)** for the complete REST reference,
including endpoints this section previously omitted entirely
(`POST /v1/verify-speaker`, `GET /v1/enroll`, `AnalyzeResponse`'s
`target_speaker` block, the `anchor` health block).

The one detail worth restating here, since it's the privacy-relevant part:
the audit log (§3 above) stores exactly the scalar features listed there —
`cm_score`, `cm_genuine_prob`, `asv_consistency`, `channel_anomaly`,
`prosody_unnatural`, `snr_db` — plus, as of the challenge-grading fix in this
revision, `challenge_passed`, `challenge_confidence`, and
`response_latency_ms` on graded-challenge events. Never raw audio, never
invertible embeddings, regardless of which endpoint produced the entry.

## 5. Configuration & Environment Variables

See **[docs/PROTOCOL.md](PROTOCOL.md) §8** for the complete, current table.
This section doesn't duplicate it — three independent copies of the same
list is exactly how it went stale in the first place (missing
`VR_ASV_BACKEND`, `VR_CM_BACKEND`, `VR_IMPOSTOR_BLOCK_BELOW`,
`VR_FUSION_W_SEMANTIC`, `GROQ_API_KEY`, `VR_ANCHOR_EMBEDDING`,
`VR_FUSION_PRESET` at various points).
