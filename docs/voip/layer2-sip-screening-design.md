# Layer-2 PSTN Screening Design: "VaniRakshak Call" Internet Calling

Status: Phase 5, design plus code skeleton only. No accounts created, no provider credentials touched, no lab-beast access used. All provider facts in this document are either cited against current provider documentation or marked TBD.

## 1. What Layer-2 is

Layer-1 (parallel workstream): in-app WebRTC calling between two VaniRakshak users, app to app over the internet.

Layer-2 (this document): screening REAL unknown callers arriving from the public phone network. The user forwards unknown cellular calls to a VaniRakshak screening number rented from a CPaaS provider. The provider answers the PSTN call, hands live audio to our screening server over a WebSocket media stream, we score the audio with the existing AASIST-L / WavLM scorer (operating threshold LLR 21.9308, bonafide about +26 and spoof about -35, verified 2026-10), and then:

- ALLOW: bridge the call to the user's app (or the user's real phone number),
- CHALLENGE: play a spoken challenge (random phrase to repeat, or a DTMF digit to press) and rescore,
- BLOCK: play a short refusal message and hang up.

Target market is India. Shortlisted providers: Exotel (Indian CPaaS, recommended first), Plivo, Twilio.

## 2. Architecture

```
  Unknown caller (PSTN)
        |
        v
  CPaaS provider (Exotel / Plivo / Twilio)
  - owns the screening number, PSTN interconnect, licensing
  - answers the call, plays/handles flow
        |
        | media stream over WSS (provider-specific format)
        v
  +-------------------------------+
  | sip-screening-server (this)   |
  |  FastAPI + WebSockets         |
  |  - provider webhook endpoints |
  |  - CallSession state machine  |
  |  - AudioBridge                |
  +-------------------------------+
        |
        | PCM 16 kHz over ws://SCORER_HOST:8000/ws/stream
        v
  Scorer backend on lab-beast (existing)
  - verdict frames carry LLR scores
```

Call control (answer, play notice, bridge, hangup) stays with the provider: we never open SIP ports ourselves. Media comes to us over the provider's WebSocket stream, and we talk back over the same stream to play spoken messages. This keeps the server simple: one public HTTPS/WSS endpoint, no raw SIP stack, no RTP port management.

How the call reaches the screening number (user side setup, manual step):
1. Rent the screening number in the provider dashboard.
2. On the user's phone, set conditional call forwarding of unknown/no-caller-ID calls to the screening number (each phone OS has per-contact forwarding rules; TBD: exact forwarding UX per Android skin).
3. Caller dials, carrier forwards, provider answers and starts the screening flow below.

## 3. Call flow, end to end

1. INBOUND RING. Caller is forwarded to the screening number. Provider receives the PSTN call.
2. ANSWER + NOTICE. Provider answers and plays the screening notice (spoken consent prompt, section 6). Audio analysis must not begin before the notice completes.
3. STREAM HANDOFF. Provider opens the WebSocket media stream to our server and begins sending caller audio frames.
4. LIVE SCORING. Our server decodes frames to 8 kHz audio, resamples to 16 kHz PCM, and pipes chunks to the scorer WebSocket. The scorer returns verdict frames with running LLR scores.
5. VERDICT. Once enough scored speech accumulates (MIN_SPEECH_SECONDS of voice, bounded by MAX_SCREEN_SECONDS total), apply the threshold policy (section 7).
6. ACTION:
   - ALLOW: provider dials a second leg to the user's phone/app (provider REST API or in-flow dial applet) and bridges. A short hand-off message plays: "This caller passed the voice screening. Connecting you now."
   - CHALLENGE: play the challenge prompt over the stream, rescore the response, then ALLOW or BLOCK.
   - BLOCK: play a refusal message ("This call did not pass verification and will not be connected."), log, hang up.
7. TEARDOWN. Stream close, status callback, write the call record (call id, caller CLI, timestamps, verdict, consent record). No audio is retained by default (section 10).

## 4. Audio path per provider

All three providers follow the same pattern: the PSTN call terminates at the provider, and the provider streams audio to our WebSocket endpoint. The formats differ. The code skeleton normalizes these behind the ProviderAdapter interface.

### 4.1 Exotel (recommended for India first)

- Configuration: in the Exotel dashboard, build an App Bazaar flow. For inbound screening the flow is: Call Start -> [Say applet with the screening notice, or have our server speak it] -> [Voicebot applet] -> [Hangup applet]. Assign the ExoPhone (the rented number) to this flow.
- Media: Exotel opens a WebSocket to the URL configured in the Voicebot applet and streams caller audio. Exotel is the WebSocket client; we host the endpoint. The Voicebot applet is bidirectional, so our server sends media frames back over the same connection to speak to the caller (screening notice, challenge, verdict messages). The Stream applet is the unidirectional variant (Exotel to us only); we use Voicebot because we need to talk back.
- Audio format: 8 kHz, 16-bit linear PCM, mono, base64-encoded in JSON frames (NOT mu-law; verify against your account's applet settings, because a mu-law/PCM mismatch produces continuous static). Events on the wire are named connected, start, media, dtmf, mark, stop. Sample rate can be raised to 16000 or 24000 via a ?sample-rate= query parameter on the Stream URL, but 8000 is the default and is what we design for. Up to 3 custom key-value parameters can be appended to the URL (256 chars total), used to carry our session id.
- Authentication: HTTP Basic with the Exotel API Key as username and API Token as password for REST calls; the applet URL itself is configured in the dashboard. India base URL is https://api.in.exotel.com (Mumbai).
- Sources: Exotel agent-stream developer guide and Connect Voice AI docs (https://github.com/exotel/agent-stream/blob/HEAD/docs/exotel-developer/connect-voice-ai.md), Exotel "one API call" reference (https://exotel.com/build/one-api-call-your-ai-bot-is-on-the-phone/), Pipecat Exotel dial-in guide (https://docs.pipecat.ai/pipecat/telephony/exotel-websockets), third-party inbound wiring notes (https://github.com/dograh-hq/dograh/blob/HEAD/docs/integrations/telephony/exotel.mdx).

### 4.2 Plivo

- Configuration: buy a number with Voice capability, create a Plivo XML application, set the number's Answer URL to our endpoint. When a call arrives, Plivo POSTs call details (including CallUUID) to the Answer URL; we respond with Plivo XML that opens the stream:
  <Stream bidirectional="true" contentType="audio/x-l16;rate=8000" keepCallAlive="true">wss://our-host/media/plivo</Stream>
- Media: Plivo opens the WebSocket with real-time audio; bidirectional mode lets us send audio back for prompts. We play the screening notice with a <Speak> element before the <Stream> element in the same answer XML.
- DTMF challenge: Plivo XML <GetInput> supports simultaneous DTMF and speech detection (inputType="dtmf speech"), which is what a press-a-digit challenge uses.
- Sources: Pipecat Plivo dial-in sequence (https://docs.pipecat.ai/pipecat/telephony/plivo-websockets), Plivo <Stream> XML shape as used in production routers (https://github.com/vobiz-ai/vobiz-xml-call-router-sarvam/blob/HEAD/README.md), Plivo GetInput DTMF+speech support (https://support.plivo.com/hc/en-us/articles/360044460151-Is-it-possible-to-gather-DTMF-and-speech-inputs-simultaneously).

### 4.3 Twilio

- Configuration: voice webhook on the Twilio number points at our endpoint. We respond with TwiML:
  <Response><Say>screening notice</Say><Connect><Stream url="wss://our-host/media/twilio"/></Connect></Response>
  Note: <Connect><Stream> is bidirectional and required for two-way audio; <Start><Stream> is listen-only and is NOT suitable here.
- Media: Twilio opens a wss WebSocket and sends JSON events: connected, then start (carries streamSid, callSid, and any custom <Parameter> values), then a steady stream of media events with base64 mu-law payloads at 8 kHz, then stop at hangup. We must decode mu-law to PCM ourselves. The Stream url accepts no query strings; pass custom key-value pairs as <Parameter> children of <Stream>, which Twilio echoes back on the start event.
- Sources: Twilio <Stream> TwiML reference (https://www.twilio.com/docs/voice/twiml/stream), Twilio Media Streams patterns (https://github.com/composite-voice/composite-voice/blob/HEAD/apps/docs/src/content/docs/guides/io/twilio-media-streams.md), webhook signature scheme (https://github.com/openai/plugins/blob/HEAD/plugins/twilio-developer-kit/skills/twilio-webhook-architecture/SKILL.md).

### 4.4 Audio normalization (our side, provider-agnostic)

Regardless of provider, the pipeline after decoding is identical:
1. Decode provider frame: base64 -> (mu-law -> PCM for Twilio; raw PCM16 for Exotel and Plivo with contentType audio/x-l16).
2. Resample 8 kHz -> 16 kHz (WavLM expects 16 kHz). Simple linear interpolation is acceptable at this stage; exact resampler is an implementation detail in AudioBridge.
3. Chunk into 1-second 16 kHz PCM16 frames and stream over the scorer WebSocket.
4. Apply a voice activity gate before sending: do not send silence to the scorer, and do not count silence toward MIN_SPEECH_SECONDS. Exact VAD choice is TBD (simple energy gate in the skeleton; a proper VAD in production).

## 5. Verdict logic

Inputs: running LLR from the scorer, seconds of scored speech, caller behavior flags (silence, hangup, DTMF).

Config (environment variables, defaults are proposals to tune live, not verified):
- OPERATING_THRESHOLD = 21.9308 (the scorer's verified operating point)
- CHALLENGE_MARGIN = 10.0 (TBD, tune live)
- MIN_SPEECH_SECONDS = 4.0 (TBD)
- MAX_SCREEN_SECONDS = 20.0 (TBD)

Decision rule, applied once scored speech reaches MIN_SPEECH_SECONDS, and forced at MAX_SCREEN_SECONDS:

- LLR >= OPERATING_THRESHOLD -> ALLOW. Bridge to the user.
- OPERATING_THRESHOLD - CHALLENGE_MARGIN <= LLR < OPERATING_THRESHOLD -> CHALLENGE. Play: "For verification, please repeat after me: blue river seven." Score the response. If the response LLR >= OPERATING_THRESHOLD -> ALLOW, else BLOCK. (Alternative challenge: "Press 5 on your keypad" using DTMF; rescore the spoken response regardless.)
- LLR < OPERATING_THRESHOLD - CHALLENGE_MARGIN -> BLOCK. Play refusal, hang up.

Rationale for the challenge band: bonafide scores cluster around +26 and spoof scores around -35, so the threshold at 21.9308 already separates the observed distributions. The challenge band exists for borderline scores caused by short, noisy, or quiet audio, where a second sample often resolves the call cleanly instead of wrongly blocking a real person.

Edge cases in scoring:
- Silent caller (no voice for MIN_SPEECH_SECONDS): treat as CHALLENGE once ("Are you still there? Please say something."), then BLOCK if still silent. Silent screening calls are a common robocall pattern; do not ALLOW silence.
- Caller hangs up mid-screening: close session, record verdict as INCOMPLETE, no bridge.
- Conflicting verdicts across chunks (LLR oscillating across the threshold): decide on the mean of the last N verdict frames, not the latest frame. N is TBD (propose 5).

## 6. Screening notice (spoken consent prompt)

Played at answer time, before any analysis. Draft script (English; Hindi version TBD):

"This call is answered by VaniRakshak, a call screening service. Your voice will be analyzed for authenticity before the call is connected. This call is not recorded. If you do not agree, please hang up now."

Requirements on the notice:
- It must complete before caller audio is sent to the scorer (consent precedes processing, DPDP Act section 6: consent must be free, specific, informed, unconditional, and unambiguous).
- The provider-specific mechanism: Twilio <Say> before <Connect><Stream>; Plivo <Speak> before <Stream>; Exotel Say applet before the Voicebot applet, or our server sends the notice as the first outbound media after the start event.
- The notice text version is logged per call as part of the consent record (section 10).

## 7. Latency budget

Target: verdict within 6 to 8 seconds of the notice completing. Budget below is a planning allocation; provider and scorer legs are TBD until measured.

| Leg | Budget | Notes |
|---|---|---|
| Provider media capture to our WebSocket (first frame) | TBD, target < 500 ms | Depends on provider; measure per provider |
| Frame decode + resample (per 1 s chunk) | < 50 ms | Local CPU, trivial |
| Scorer round trip (send chunk, receive verdict frame) | TBD, target < 800 ms per chunk | Depends on lab-beast GPU load; measure |
| Accumulate MIN_SPEECH_SECONDS of voice | 4.0 s (config) | Dominant term; unavoidable |
| Verdict decision + prompt playback start | < 500 ms | TTS or pre-recorded audio; pre-recorded is faster |
| Total, typical | ~5 to 6 s | Notice duration (~6 s spoken) runs before this clock starts |
| Total, worst case (challenge path) | ~20 s | Bounded by MAX_SCREEN_SECONDS |

Design consequences: keep the notice pre-recorded (not live TTS) to remove a variable; score overlapping 1-second windows so the verdict is ready the moment MIN_SPEECH_SECONDS is reached; bridge via provider dial-leg (one REST call) rather than any app-side handshake in the ALLOW path.

## 8. Failure modes

- Caller hangs up during screening: the provider sends stop (Twilio/Exotel) or closes the stream. Action: mark session INCOMPLETE, close scorer pipe, write call record. Never attempt a bridge after a stop event.
- Caller hangs up during challenge: same as above; do not retry.
- Scorer unreachable (lab-beast down, WS refused): fail closed. Play: "Screening is temporarily unavailable. Your call cannot be verified right now." Hang up. Do NOT bridge unverified calls; bridging without a score defeats the product. Alert the operator (log + optional push; mechanism TBD).
- Scorer slow (verdict frames stop arriving): if no verdict for VERDICT_TIMEOUT seconds (propose 10 s, TBD), treat as scorer unreachable.
- Provider webhook retries: providers retry webhooks on non-2xx or timeouts. Every webhook handler must be idempotent: key sessions by provider call id (CallSid / CallUUID / Exotel Sid) and ignore duplicate events for a session that already exists.
- Duplicate stream connections: if a second start event arrives for the same call id, keep the first session and close the second stream.
- Stream connect failure (provider cannot reach our WSS): provider-side behavior varies (call drops or falls through to the next flow applet). Mitigation: health monitoring of the WSS endpoint, TLS cert expiry alerts, and a provider-side fallback applet that plays "service unavailable" instead of dead air. Exact fallback config is provider-specific and TBD at setup time.
- Silent caller: handled in section 5.
- Webhook signature mismatch: reject with 403, log, do not create a session. Never process unsigned webhooks in production.
- Clock of verdict frames vs audio: if the scorer echoes timestamps, align by timestamp; otherwise align by chunk sequence number. Protocol detail TBD against the actual /ws/stream implementation.

## 9. Call state machine

One CallSession per provider call id. Transitions are guarded: only the listed transitions are legal; anything else is logged and ignored.

```
                    +-----------+
                    |  RINGING  |  webhook received, session created
                    +-----+-----+
                          | answer / stream start
                          v
                    +-----------+
                    |  NOTICE   |  consent prompt playing, NOT scoring
                    +-----+-----+
                          | notice complete
                          v
                    +-----------+
                    | SCREENING |<---------------------+
                    +-----+-----+                      |
          +---------------+---------------+           | challenge response
          |               |               |           | received, rescoring
          v               v               v           |
   +-----------+   +-----------+   +-----------+      |
   |  ALLOW    |   | CHALLENGE |-->| RESCORING |- -----+
   +-----+-----+   +-----------+   +-----+-----+
         |                             |
         v                    +--------+--------+
   +-----------+             v                 v
   | BRIDGING  |      +-----------+       +-----------+
   +-----+-----+      | ALLOW (2) |       |   BLOCK   |
         |            +-----+-----+       +-----+-----+
         v                  |                   |
   +-----------+           v                   v
   |  BRIDGED  |     (to BRIDGING)       +-----------+
   +-----+-----+                         | REFUSING  |  refusal message playing
         |                               +-----+-----+
         v                                     |
   +-----------+  caller/provider hangup       v
   |   ENDED   |<------------------------+-----------+
   +-----------+                         |
                                         v
                                  +-----------+
                                  |   ENDED   |
                                  +-----------+

  Any state --(stop event / hangup / error)--> ENDED (terminal)
  Any state --(scorer unreachable)--> FAILSAFE --> ENDED
```

State notes:
- NOTICE: caller audio arriving during the notice is buffered, not scored.
- SCREENING: verdict evaluated on each verdict frame once MIN_SPEECH_SECONDS reached.
- CHALLENGE: outbound challenge prompt playing; caller audio not scored until prompt ends.
- RESCORING: scoring the challenge response only.
- BRIDGING: second leg dial in progress; caller hears hold audio (provider default).
- BRIDGED: both legs connected; our server leaves the media path (provider bridges directly).
- FAILSAFE: spoken "unavailable" message, then ENDED.
- ENDED: terminal; call record written.

## 10. Security and privacy

- Webhook signature verification, per provider, on every inbound webhook:
  - Twilio: X-Twilio-Signature header, base64 of HMAC-SHA1 over (full URL + sorted POST params), key is the Twilio Auth Token. Implement with the official twilio SDK RequestValidator where possible; the skeleton implements the documented algorithm directly. Source: https://github.com/openai/plugins/blob/HEAD/plugins/twilio-developer-kit/skills/twilio-webhook-architecture/SKILL.md
  - Plivo: X-Plivo-Signature-V2 header with X-Plivo-Signature-V2-Nonce, HMAC-SHA256 over (base URL + nonce), key is the Plivo Auth Token. Source: Plivo signature validation docs as referenced in https://github.com/plivo/plivo-node/issues/338
  - Exotel: applet URL resolution can carry HTTP Basic auth; the status-callback signature scheme is TBD and must be confirmed against current Exotel docs before production.
- Secrets live in environment variables only. No secrets in code, logs, or the repo. Log call metadata (numbers, timestamps, verdict), never audio content or auth tokens.
- TLS everywhere: webhooks over HTTPS, media over WSS. Reject plain ws/http in production config.
- Idempotency keys on provider call ids to survive webhook retries without double sessions or double bridges.
- Privacy: audio is streamed for scoring and not retained by default. If any debugging capture is ever enabled, it requires the same consent notice to say so, plus a retention and deletion policy (see the setup guide's regulatory checklist). Consent records (notice version, timestamp, caller CLI) are kept as the audit trail.

## 11. What is intentionally out of scope for Phase 5

- App-to-app data-channel bridge after ALLOW (Phase 6 option). Phase 5 bridges via provider dial-leg to the user's phone number.
- Real TTS integration for prompts: prompts are pre-recorded audio files or provider <Say>/<Speak>; live TTS is a later optimization.
- Production hardening: rate limiting per caller, blocklist/allowlist management UI, analytics dashboard, multi-number support.
- Direct SIP trunking (we use provider WebSocket streams exclusively; no SIP ports are opened).

## 12. TBD list (design level)

1. Exact Exotel Voicebot event payload fields beyond event names (confirm against current AgentStream docs).
2. Plivo stream lifecycle callback fields (confirm against current Plivo XML docs).
3. CHALLENGE_MARGIN, MIN_SPEECH_SECONDS, MAX_SCREEN_SECONDS tuning values (tune with live test calls).
4. Scorer /ws/stream frame protocol (request/response schema) as implemented on lab-beast.
5. Challenge aggregation rule (proposed: mean of last 5 verdict frames).
6. Provider media leg latency (measure per provider in testing).
7. Conditional call forwarding UX for unknown callers on the user's Android phone.
8. Hindi and other Indian language versions of the screening notice.
