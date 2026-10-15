"""FastAPI app: provider webhook endpoints + media WebSocket handlers.

Endpoints:
  GET  /health                        liveness, no credentials needed
  POST /webhooks/exotel/resolve        Exotel applet URL resolver, returns {"url": wss://...}
  POST /webhooks/plivo/answer          Plivo Answer URL, returns Plivo XML with <Stream>
  POST /webhooks/twilio/voice          Twilio voice webhook, returns TwiML
  WS   /media/{provider}               provider media stream (exotel|plivo|twilio)

The media handler drives one CallSession through the state machine:
RINGING -> NOTICE -> SCREENING -> ALLOW/CHALLENGE/BLOCK -> bridge or hangup.
Outbound actions (dial second leg, hangup) go through provider REST APIs and
are stubbed as functions raising NotImplementedError until a provider is
chosen and credentials are configured.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from .bridge import AudioBridge, ScorerUnreachable
from .config import settings
from .providers import StreamEvent, get_adapter, pcm16_8k_to_16k
from .security import verify_plivo_signature, verify_twilio_signature
from .session import CallSession, CallState, IllegalTransition

log = logging.getLogger("sip_screening")

NOTICE_TEXT = (
    "This call is answered by VaniRakshak, a call screening service. "
    "Your voice will be analyzed for authenticity before the call is connected. "
    "This call is not recorded. If you do not agree, please hang up now."
)
CHALLENGE_TEXT = (
    "For verification, please repeat after me: blue river seven."
)
REFUSAL_TEXT = (
    "This call did not pass verification and will not be connected. Goodbye."
)
UNAVAILABLE_TEXT = (
    "Screening is temporarily unavailable. "
    "Your call cannot be verified right now. Goodbye."
)

app = FastAPI(title="VaniRakshak Layer-2 SIP screening (skeleton)")

# provider_call_id -> CallSession. In production this must be a shared store
# (Redis) so webhook retries and reconnects hit the same session.
_sessions: dict[str, CallSession] = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": settings.PROVIDER,
            "scorer_ws": settings.scorer_ws_url}


def _full_url(request: Request) -> str:
    # Honor proxy headers so Twilio's signature check sees the public URL.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    return f"{proto}://{host}{request.url.path}"


@app.post("/webhooks/exotel/resolve")
async def exotel_resolve(request: Request) -> Response:
    """Exotel Voicebot-applet URL resolver: returns the WSS media URL.

    Exotel verification scheme is TBD (see security.py); this endpoint
    requires HTTP Basic auth credentials configured in the applet until
    the signed scheme is confirmed.
    """
    form = dict(await request.form())
    call_id, caller, called = get_adapter("exotel").extract_call_info(form)
    session = _get_or_create_session("exotel", call_id, caller, called)
    media_url = f"{settings.media_ws_base}/media/exotel?session={session.session_id}"
    return Response(content='{"url": "%s"}' % media_url,
                    media_type="application/json")


@app.post("/webhooks/plivo/answer")
async def plivo_answer(request: Request) -> Response:
    form = dict(await request.form())
    signature = request.headers.get("X-Plivo-Signature-V2", "")
    nonce = request.headers.get("X-Plivo-Signature-V2-Nonce", "")
    if settings.PLIVO_AUTH_TOKEN and not verify_plivo_signature(
        _full_url(request).split("?")[0], nonce, signature,
        settings.PLIVO_AUTH_TOKEN,
    ):
        log.warning("plivo webhook signature mismatch")
        return Response(status_code=403)
    call_id, caller, called = get_adapter("plivo").extract_call_info(form)
    session = _get_or_create_session("plivo", call_id, caller, called)
    content_type, body = get_adapter("plivo").answer_body(
        f"{settings.media_ws_base}/media/plivo?session={session.session_id}",
        NOTICE_TEXT,
    )
    return Response(content=body, media_type=content_type)


@app.post("/webhooks/twilio/voice")
async def twilio_voice(request: Request) -> Response:
    form = dict(await request.form())
    signature = request.headers.get("X-Twilio-Signature", "")
    if settings.TWILIO_AUTH_TOKEN and not verify_twilio_signature(
        _full_url(request), form, signature, settings.TWILIO_AUTH_TOKEN
    ):
        log.warning("twilio webhook signature mismatch")
        return Response(status_code=403)
    call_id, caller, called = get_adapter("twilio").extract_call_info(form)
    session = _get_or_create_session("twilio", call_id, caller, called)
    content_type, body = get_adapter("twilio").answer_body(
        f"{settings.media_ws_base}/media/twilio?session={session.session_id}",
        NOTICE_TEXT,
    )
    return Response(content=body, media_type=content_type)


def _get_or_create_session(provider: str, call_id: str, caller: str,
                           called: str) -> CallSession:
    """Idempotent session creation: provider webhook retries reuse the
    existing session instead of creating a duplicate."""
    key = f"{provider}:{call_id or 'unknown'}"
    session = _sessions.get(key)
    if session is None or session.is_terminal():
        session = CallSession(provider=provider, provider_call_id=call_id,
                              caller_cli=caller, called_number=called,
                              notice_version=settings.NOTICE_VERSION)
        _sessions[key] = session
        log.info("new session %s for %s call %s", session.session_id,
                 provider, call_id)
    return session


@app.websocket("/media/{provider}")
async def media_stream(websocket: WebSocket, provider: str):
    await websocket.accept()
    adapter = get_adapter(provider)
    session_id = websocket.query_params.get("session", "")
    session = _find_session(session_id)
    if session is None:
        log.warning("media stream with unknown session %s", session_id)
        await websocket.close()
        return

    pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def pcm_source():
        while True:
            chunk = await pcm_queue.get()
            if chunk is None:
                return
            yield chunk

    scorer_task = asyncio.create_task(_score_loop(session, pcm_source()))

    try:
        _safe_transition(session, CallState.NOTICE, "stream opened")
        # NOTICE: buffer caller audio, do NOT score until the notice is done.
        # The notice itself is played by the provider (<Say>/<Speak>/applet),
        # so here we just wait for the start event before scoring.
        scoring = False
        async for raw in websocket.iter_text():
            event = adapter.parse_media_message(raw)
            if event.kind == "start":
                scoring = True
                _safe_transition(session, CallState.SCREENING, "notice complete")
            elif event.kind == "media" and scoring:
                pcm_queue.put_nowait(pcm16_8k_to_16k(event.pcm16_8k))
            elif event.kind == "dtmf":
                log.info("session %s dtmf %s", session.session_id,
                         event.dtmf_digit)
            elif event.kind == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        pcm_queue.put_nowait(None)
        await scorer_task
        _safe_transition(session, CallState.ENDED, "stream closed")
        log.info("session %s ended, verdict=%s",
                 session.session_id,
                 session.verdict.decision if session.verdict else "incomplete")


def _find_session(session_id: str) -> CallSession | None:
    for session in _sessions.values():
        if session.session_id == session_id and not session.is_terminal():
            return session
    return None


def _safe_transition(session: CallSession, state: CallState, note: str) -> None:
    try:
        session.transition(state, note)
    except IllegalTransition as exc:
        log.warning("illegal transition ignored: %s", exc)


async def _score_loop(session: CallSession, pcm_source) -> None:
    """Drive scoring and the verdict policy for one call."""
    bridge = AudioBridge(settings.scorer_ws_url)
    started = time.time()
    try:
        async for frame in bridge.pipe_to_scorer(pcm_source()):
            if session.state is not CallState.SCREENING:
                continue
            session.record_llr(frame.llr, frame.scored_speech_seconds)
            verdict = session.decide(settings.OPERATING_THRESHOLD,
                                     settings.CHALLENGE_MARGIN,
                                     settings.MIN_SPEECH_SECONDS)
            if verdict is not None:
                await _apply_verdict(session, verdict)
                return
            if time.time() - started > settings.MAX_SCREEN_SECONDS:
                # Force a decision on whatever we have.
                verdict = session.decide(settings.OPERATING_THRESHOLD,
                                         settings.CHALLENGE_MARGIN, 0.0)
                if verdict is None:
                    verdict = session.decide(
                        settings.OPERATING_THRESHOLD,
                        settings.CHALLENGE_MARGIN, 0.0) or _block_verdict(session)
                await _apply_verdict(session, verdict)
                return
    except ScorerUnreachable as exc:
        log.error("scorer unreachable for session %s: %s",
                  session.session_id, exc)
        _safe_transition(session, CallState.FAILSAFE, "scorer unreachable")
        await _speak(session, UNAVAILABLE_TEXT)


def _block_verdict(session: CallSession):
    from .session import Verdict
    return Verdict("block", float("-inf"), session.scored_speech_seconds,
                   "no scored speech within the screening window")


async def _apply_verdict(session: CallSession, verdict) -> None:
    session.verdict = verdict
    log.info("session %s verdict %s (llr=%.2f)", session.session_id,
             verdict.decision, verdict.final_llr)
    if verdict.decision == "allow":
        _safe_transition(session, CallState.ALLOW, verdict.reason)
        await _bridge_to_user(session)
    elif verdict.decision == "challenge":
        _safe_transition(session, CallState.CHALLENGE, verdict.reason)
        await _speak(session, CHALLENGE_TEXT)
        # Challenge-response rescoring is a Phase 6 implementation step:
        # after the prompt ends, move to RESCORING and run decide() again
        # on fresh frames, then challenge_decide(). Stubbed here.
        raise NotImplementedError(
            "challenge-response rescoring loop is stubbed (Phase 6)")
    else:
        _safe_transition(session, CallState.BLOCK, verdict.reason)
        _safe_transition(session, CallState.REFUSING, "playing refusal")
        await _speak(session, REFUSAL_TEXT)


async def _bridge_to_user(session: CallSession) -> None:
    """Place the second leg to the user's phone and bridge. Stubbed: each
    provider dials the second leg through its own REST API (Exotel Connect,
    Plivo calls.create, Twilio calls.create), which needs real credentials
    and is implemented when a provider is chosen."""
    _safe_transition(session, CallState.BRIDGING, "dialing user leg")
    raise NotImplementedError(
        "provider second-leg dial is stubbed until a provider is chosen")


async def _speak(session: CallSession, text: str) -> None:
    """Play a spoken message to the caller over the media stream. Stubbed:
    implemented per provider by sending media/mark frames (Exotel/Twilio)
    or stream audio (Plivo), using pre-recorded notice audio files."""
    raise NotImplementedError(
        "outbound media playback is stubbed; wire pre-recorded audio here")
