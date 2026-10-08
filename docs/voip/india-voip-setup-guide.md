# India VoIP Setup Guide: Layer-2 PSTN Screening

Operational checklist for standing up the VaniRakshak Call screening number
in India. This is a Phase 5 planning document: follow it when the build
moves to Phase 6. Every provider fact below is cited against a current
source or marked TBD. Prices are never invented: where a price cannot be
verified, the guide says to confirm it on the provider's pricing page.

> Note: this guide is operational, not legal advice. Have a lawyer review
> the regulatory checklist before handling real caller traffic.

## 1. Choose the provider

Recommended order for an India-first rollout:

1. Exotel. Indian CPaaS with Mumbai infrastructure, INR billing, and native
   WebSocket voice streaming (AgentStream / Voicebot applet). Lowest
   regulatory friction for Indian numbers.
2. Plivo. Strong Voice API and XML flow control; verify India number
   availability and compliance requirements with Plivo sales for your use.
3. Twilio. Mature Media Streams, but confirm India-specific voice
   availability, number types, and regulatory terms with Twilio before
   committing; inbound international-call routing rules for India differ
   from other markets.

TBD: confirm India inbound number availability and any market-specific
restrictions directly with Plivo and Twilio before purchase.

## 2. Accounts: signup checklist

### Exotel
1. Register at the Exotel India signup page (my.in.exotel.com). Sign up as
   the business entity that will own the screening number.
2. Verify email and mobile when prompted.
3. Complete KYC in the dashboard under My Account, KYC Docs tab, before
   requesting phone numbers or voice streaming. The company details on the
   Company Info page must match the uploaded documents exactly.
4. Buy an ExoPhone (virtual number) from the dashboard.
5. Ask Exotel support to enable voice streaming / the Voicebot applet on
   the account if it is not already enabled, stating the AI voice use case.
   Source for this flow: https://github.com/yashrastogi069-dev/voice-agent/blob/HEAD/EXOTEL_BEGINNER_SETUP.md
6. Collect: Account SID, API Key, API Token, ExoPhone number.
7. Build the App Bazaar flow: Call Start -> Say applet (screening notice)
   -> Voicebot applet (WSS URL of this server) -> Hangup applet. Assign the
   ExoPhone to the flow. Source:
   https://docs.pipecat.ai/pipecat/telephony/exotel-websockets

### Plivo
1. Create a Plivo account and complete any required identity verification.
2. Collect: Auth ID and Auth Token (used both for REST calls and webhook
   signature verification).
3. Buy a phone number with Voice capability in the Plivo console.
4. Create a Plivo XML application and set the number's Answer URL to
   https://<server>/webhooks/plivo/answer. The server responds with XML
   containing <Speak> (notice) and a bidirectional <Stream> to
   wss://<server>/media/plivo. Source:
   https://docs.pipecat.ai/pipecat/telephony/plivo-websockets

### Twilio
1. Create a Twilio account (trial available) and complete verification.
2. Collect: Account SID and Auth Token (the Auth Token is the key for
   X-Twilio-Signature verification; prefer API keys for outbound REST).
3. Buy a phone number; set its voice webhook to
   https://<server>/webhooks/twilio/voice. The server responds with TwiML:
   <Say> notice, then <Connect><Stream url="wss://<server>/media/twilio"/>.
   Source: https://www.twilio.com/docs/voice/twiml/stream
4. Webhook URL must be public HTTPS; the Stream URL must be wss (TLS is
   the only supported protocol for Media Streams).

TBD: trial-account limits on inbound calling and streaming for each
provider; verify in the provider console before the first live test.

## 3. KYC documents typically required in India

Indian telecom regulation requires verified identity and address for the
account holder before any number goes live. This is a regulatory gate, not
a provider policy that can be waived.

Exotel's published KYC list (varies by entity type; confirm the current
list in the dashboard):
- Registered company: certificate of incorporation (or MSME / DIPP / FSSAI
  certificate), company PAN card, business address proof (GST certificate,
  utility bill, rental agreement, or similar), passport-size photograph of
  a director or authorized signatory.
- Partnership / LLP: partnership deed or LLP incorporation certificate,
  PAN, address proof, partner photograph.
- NGO / trust / society: registration documents per the support page.
Source: https://support.exotel.com/support/solutions/articles/35760-where-do-i-upload-my-kyc-verification-docs-and-what-documents-qualify-for-kyc-
General India context: https://edesy.in/blog/indian-virtual-number-from-abroad-nri-guide-2026

Practical notes:
- Upload formats accepted: png, gif, jpeg, pdf (per Exotel's support page).
- Company details in the dashboard must match the documents; bills and
  refunds are issued in that company name.
- KYC/CAF approval can take days; start it before any other setup step.

## 4. Virtual number types and cost components

Number types commonly offered:
- Long code (10-digit virtual mobile/landline-like number): the natural
  choice for a screening number users forward calls to.
- Toll-free (1800 series): recognizable, but the called party (us) pays
  higher per-minute rates; unnecessary for a forwarding target.
- Geographic STD numbers in specific circles.

Cost components to budget (confirm current pricing on each provider's
pricing page; do not rely on third-party price quotes):
1. Number rental: monthly per-number charge.
2. Inbound per-minute charges on the screening number.
3. Outbound per-minute charges for the second leg when bridging an allowed
   call to the user's phone (this roughly doubles per-minute cost on every
   allowed call).
4. Recording storage: only if recording is ever enabled (it is off by
   default in this design); storage and retrieval fees apply per provider.
5. SMS charges if the flow ever sends SMS (not in the Phase 5 design).
6. Taxes (GST) on top of listed rates.

Pricing pages to check at build time: Exotel pricing, Plivo pricing,
Twilio voice pricing for India. Prices change; re-confirm before budgeting.

## 5. Server requirements

Because all three providers stream media over WebSockets, we do NOT need
to open SIP (5060) or RTP ports, and we do NOT run a SIP stack:

- One public HTTPS endpoint for webhooks (port 443).
- One public WSS endpoint for media streams (can share port 443 with the
  webhooks behind the same TLS termination).
- Valid TLS certificate from a public CA. Twilio requires wss for Media
  Streams; Exotel's applet accepts ws:// or wss:// but production must use
  wss://.
- The webhook URL must be reachable from the provider's cloud; test with
  the provider's webhook debugger before pointing a live number at it.
- Outbound HTTPS from the server to api.in.exotel.com (or the chosen
  provider's API base) for second-leg dials and status callbacks.
- No inbound SIP/RTP ports needed. If the design ever moves to direct SIP
  trunking instead of provider media streams, the port plan is TBD then.

Scaling note for Phase 6: the skeleton keeps sessions in memory. Run one
process per server in testing; move sessions to a shared store before any
multi-instance deployment.

## 6. Regulatory checklist

The licensed CPaaS provider carries interconnect and telecom licensing
compliance for the PSTN leg. The following are OUR responsibilities and
must be implemented before handling real calls:

1. Spoken screening notice before analysis. Play the consent prompt
   (design doc section 6) at answer time, before any caller audio is sent
   to the scorer. The notice text version is logged per call.
2. DPDP Act 2023 notice and consent. Voice is personal data. Section 5
   requires a clear notice of what is processed and why; Section 6
   requires consent that is free, specific, informed, unconditional, and
   unambiguous, given by explicit affirmative action. The screening notice
   plus the caller's choice to stay on the line is the consent mechanism;
   store a consent record per call (caller CLI, timestamp, notice
   version). Sources:
   http://www.forbesindia.com/blog/legalese/decoding-the-digital-personal-data-protection-act-2023-329553.html
   and https://github.com/vaibhav-aiml/ai-medical-voice-agent/blob/HEAD/README-DPDP.md
3. Consent withdrawal. If a caller objects mid-call, stop processing and
   end the screening path. Document how a caller can request deletion of
   their data.
4. Data retention and deletion policy. Define: what is kept (call
   metadata, verdict, consent record), what is never kept (audio by
   default), how long records live, and how deletion requests are honored.
5. No call recording without the prompt. Recording stays OFF unless the
   notice explicitly says the call is recorded, the caller consents, and
   the consent is logged. A notification alone is not enough; the consent
   record must exist. Source:
   https://frejun.com/call-recording-setup-compliance-india/
6. DND discipline. The screening number is a defensive inbound service,
   not a marketing channel. If the product ever sends outbound promotional
   SMS or calls, DLT registration and consent rules apply separately.

## 7. Go-live test checklist

1. Signature verification: send a forged webhook (wrong signature) and
   confirm a 403 with no session created, for each active provider.
2. Notice ordering: call the screening number, confirm the full notice
   plays before any scoring verdict is possible (check server logs for
   NOTICE -> SCREENING transition timing).
3. Happy path: known-good caller passes screening and gets bridged.
4. Challenge path: borderline audio triggers the challenge prompt.
5. Block path: spoofed sample is refused and hung up on.
6. Hangup mid-screening: caller hangs up during the notice and during
   scoring; confirm INCOMPLETE records and no bridge attempts.
7. Scorer down: stop the scorer, call in, confirm the failsafe message
   plays and the call is NOT bridged.
8. Webhook retry storm: replay the same provider webhook five times;
   confirm exactly one session and one verdict.
9. Silent caller: stay silent for the full window; confirm challenge then
   block, never allow.
10. TLS/cert check: confirm wss connects and the certificate renews
    without breaking the media URL (use a stable hostname, not an IP).

## 8. TBD items carried from the design doc

- Exotel webhook/status-callback signature scheme (confirm docs, then
  implement verify_exotel).
- Plivo stream lifecycle callback field details.
- Tuned values for CHALLENGE_MARGIN, MIN_SPEECH_SECONDS,
  MAX_SCREEN_SECONDS.
- Scorer /ws/stream frame schema on lab-beast (needed before _send_chunk
  and _parse_verdict go live).
- Hindi and other Indian language notice scripts.
- Conditional call-forwarding UX on the user's Android phone for routing
  unknown callers to the screening number.
