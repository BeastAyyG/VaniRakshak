"""Provider-agnostic CallSession state machine.

One CallSession exists per provider call id (Twilio CallSid, Plivo CallUUID,
Exotel call Sid). Only the transitions in TRANSITIONS are legal; anything
else is rejected and logged so a stray provider event can never push a call
into an impossible state (for example bridging a call that already ended).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class CallState(str, Enum):
    RINGING = "ringing"        # webhook received, session created
    NOTICE = "notice"          # consent prompt playing, NOT scoring
    SCREENING = "screening"    # live audio being scored
    CHALLENGE = "challenge"    # challenge prompt playing
    RESCORING = "rescoring"    # scoring the challenge response
    ALLOW = "allow"            # verdict: allow
    BLOCK = "block"            # verdict: block
    BRIDGING = "bridging"      # second leg dial in progress
    BRIDGED = "bridged"        # caller connected to the user
    REFUSING = "refusing"      # refusal message playing
    FAILSAFE = "failsafe"      # scorer unavailable; spoken message, then end
    ENDED = "ended"            # terminal


# Legal transitions: current state -> set of allowed next states.
TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.RINGING: {CallState.NOTICE, CallState.ENDED},
    CallState.NOTICE: {CallState.SCREENING, CallState.ENDED, CallState.FAILSAFE},
    CallState.SCREENING: {
        CallState.ALLOW, CallState.CHALLENGE, CallState.BLOCK,
        CallState.ENDED, CallState.FAILSAFE,
    },
    CallState.CHALLENGE: {CallState.RESCORING, CallState.ENDED},
    CallState.RESCORING: {CallState.ALLOW, CallState.BLOCK, CallState.ENDED, CallState.FAILSAFE},
    CallState.ALLOW: {CallState.BRIDGING, CallState.ENDED},
    CallState.BLOCK: {CallState.REFUSING, CallState.ENDED},
    CallState.BRIDGING: {CallState.BRIDGED, CallState.ENDED},
    CallState.BRIDGED: {CallState.ENDED},
    CallState.REFUSING: {CallState.ENDED},
    CallState.FAILSAFE: {CallState.ENDED},
    CallState.ENDED: set(),
}


class IllegalTransition(Exception):
    """Raised when a provider event tries an impossible state change."""


@dataclass
class Verdict:
    decision: str          # "allow" | "challenge" | "block"
    final_llr: float
    scored_speech_seconds: float
    reason: str = ""


@dataclass
class CallSession:
    """Mutable per-call state. Created by the webhook handler, driven by
    media-stream events and verdict frames from the scorer."""

    provider: str
    provider_call_id: str
    caller_cli: str = ""
    called_number: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: CallState = CallState.RINGING
    created_at: float = field(default_factory=time.time)
    notice_version: str = ""
    scored_speech_seconds: float = 0.0
    llr_history: list[float] = field(default_factory=list)
    verdict: Verdict | None = None
    history: list[tuple[float, str, str]] = field(default_factory=list)

    def transition(self, next_state: CallState, note: str = "") -> None:
        allowed = TRANSITIONS[self.state]
        if next_state not in allowed:
            raise IllegalTransition(
                f"session {self.session_id}: {self.state} -> {next_state} "
                f"is not legal (note: {note})"
            )
        self.history.append((time.time(), self.state.value, next_state.value))
        self.state = next_state

    def record_llr(self, llr: float, speech_seconds: float) -> None:
        """Record one verdict frame from the scorer."""
        self.llr_history.append(llr)
        self.scored_speech_seconds += speech_seconds

    @property
    def last_llr(self) -> float | None:
        return self.llr_history[-1] if self.llr_history else None

    def decide(self, operating_threshold: float, challenge_margin: float,
               min_speech_seconds: float) -> Verdict | None:
        """Apply the threshold policy from the design doc (section 5).

        Returns None while there is not enough scored speech to decide.
        """
        if self.scored_speech_seconds < min_speech_seconds or not self.llr_history:
            return None
        llr = self.llr_history[-1]
        if llr >= operating_threshold:
            return Verdict("allow", llr, self.scored_speech_seconds,
                           "LLR at or above operating threshold")
        if llr >= operating_threshold - challenge_margin:
            return Verdict("challenge", llr, self.scored_speech_seconds,
                           "LLR in challenge band below threshold")
        return Verdict("block", llr, self.scored_speech_seconds,
                       "LLR below challenge band")

    def challenge_decide(self, operating_threshold: float,
                         aggregation_frames: int) -> Verdict:
        """Decide on the challenge response using the mean of the last N
        verdict frames (proposal from the design doc; tune live)."""
        frames = self.llr_history[-aggregation_frames:] or self.llr_history
        mean_llr = sum(frames) / len(frames) if frames else float("-inf")
        if mean_llr >= operating_threshold:
            return Verdict("allow", mean_llr, self.scored_speech_seconds,
                           "challenge response passed")
        return Verdict("block", mean_llr, self.scored_speech_seconds,
                       "challenge response failed")

    def is_terminal(self) -> bool:
        return self.state is CallState.ENDED
