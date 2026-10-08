# sip-screening-server (Phase 5 skeleton)

Layer-2 PSTN screening for VaniRakshak Call: a CPaaS provider answers an
inbound PSTN call on a rented screening number, streams caller audio to this
server over WebSocket, and this server pipes the audio to the scorer
WebSocket on lab-beast, applies the verdict policy, and directs the provider
(bridge, challenge, or block).

## What is real vs stubbed

REAL (implemented, tested by import and unit logic):
- FastAPI app with webhook endpoints for Exotel, Plivo, and Twilio.
- ProviderAdapter interface with three adapters: Exotel (8 kHz PCM16),
  Plivo (Plivo XML with bidirectional Stream, L16 8 kHz), Twilio (TwiML
  with Connect/Stream, mu-law decode to PCM16).
- CallSession state machine with guarded transitions (illegal transitions
  raise and are logged, never applied).
- Threshold verdict policy: allow / challenge / block around the verified
  operating point LLR 21.9308, plus challenge-response aggregation.
- Twilio and Plivo webhook signature verification (HMAC-SHA1 and
  HMAC-SHA256 respectively, per each provider's documented scheme).
- AudioBridge: 8 kHz to 16 kHz upsampling, silence gating, chunking, and
  the scorer WebSocket plumbing. The scorer frame schema is a documented
  best guess and is marked TBD.

STUBBED (raises NotImplementedError with a note, to be built in Phase 6):
- Outbound media playback (_speak): sending pre-recorded notice, challenge,
  and refusal audio back over each provider's stream.
- Second-leg dial (_bridge_to_user): provider REST call to bridge allowed
  callers to the user's phone.
- Challenge-response rescoring loop (CHALLENGE -> RESCORING states).
- Exotel webhook signature verification: the scheme is TBD against current
  Exotel docs, so verify_exotel fails closed.
- Session store is an in-memory dict; production needs a shared store.

## Run

No provider credentials are needed to import the package or run the health
check.

```bash
cd sip-screening-server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn sip_screening_server.app:app --host 0.0.0.0 --port 8080
curl localhost:8080/health
```

## Layout

- sip_screening_server/app.py: FastAPI app, webhook endpoints, media
  WebSocket handler, verdict application.
- sip_screening_server/session.py: CallSession state machine and verdict
  policy.
- sip_screening_server/providers.py: ProviderAdapter ABC plus Exotel,
  Plivo, and Twilio adapters and audio decoding helpers.
- sip_screening_server/bridge.py: AudioBridge with pipe_to_scorer.
- sip_screening_server/security.py: webhook signature verification.
- sip_screening_server/config.py: environment-variable configuration.

## Syntax check

```bash
python -m py_compile $(find sip_screening_server -name "*.py")
```
