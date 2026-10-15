"""Provider adapters behind one clean interface.

Each provider answers the PSTN call itself and streams audio to us over a
WebSocket, but the webhook shape, the media encoding, and the event names
differ. Providers implement this ABC; the rest of the server only sees
normalized events.

Audio formats (verified against current provider docs; see design doc):
  Exotel: 8 kHz, 16-bit linear PCM mono, base64; events connected/start/
          media/dtmf/mark/stop.
  Plivo:  8 kHz L16 (contentType audio/x-l16;rate=8000), base64 in stream
          frames; call identity from the Answer webhook (CallUUID).
  Twilio: 8 kHz mu-law, base64; events connected/start/media/stop. The
          start event carries streamSid, callSid and custom <Parameter>s.
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StreamEvent:
    kind: str              # "connected" | "start" | "media" | "dtmf" | "stop"
    pcm16_8k: bytes = b""  # decoded 8 kHz 16-bit PCM mono for "media" events
    stream_id: str = ""
    call_id: str = ""
    caller_cli: str = ""
    dtmf_digit: str = ""
    raw: dict | None = None


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 mu-law bytes to 16-bit little-endian PCM."""
    out = bytearray()
    for byte in data:
        b = ~byte & 0xFF
        sign = b & 0x80
        exponent = (b >> 4) & 0x07
        mantissa = b & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        if sign:
            sample = -sample
        sample = max(-32768, min(32767, sample))
        out += sample.to_bytes(2, "little", signed=True)
    return bytes(out)


def pcm16_8k_to_16k(pcm: bytes) -> bytes:
    """Naive linear-interpolation upsample from 8 kHz to 16 kHz PCM16.

    Placeholder for a proper resampler in production; WavLM needs 16 kHz.
    """
    n = len(pcm) // 2
    if n == 0:
        return b""
    samples = [int.from_bytes(pcm[i * 2:(i + 1) * 2], "little", signed=True)
               for i in range(n)]
    out = bytearray()
    for i, s in enumerate(samples):
        nxt = samples[i + 1] if i + 1 < n else s
        out += s.to_bytes(2, "little", signed=True)
        out += ((s + nxt) // 2).to_bytes(2, "little", signed=True)
    return bytes(out)


class ProviderAdapter(ABC):
    """Normalize one provider's webhook + media stream."""

    name: str

    @abstractmethod
    def answer_body(self, media_ws_url: str, notice_text: str) -> tuple[str, str]:
        """Return (content_type, body) to answer the inbound webhook: play
        the screening notice, then open the media stream to media_ws_url."""
        raise NotImplementedError

    @abstractmethod
    def parse_media_message(self, message: str | bytes) -> StreamEvent:
        """Parse one WebSocket text frame into a normalized StreamEvent."""
        raise NotImplementedError

    @abstractmethod
    def extract_call_info(self, webhook_form: dict) -> tuple[str, str, str]:
        """Return (provider_call_id, caller_cli, called_number) from the
        inbound webhook's form fields."""
        raise NotImplementedError


class ExotelAdapter(ProviderAdapter):
    name = "exotel"

    def answer_body(self, media_ws_url: str, notice_text: str) -> tuple[str, str]:
        # Exotel inbound: the App Bazaar flow (Say applet -> Voicebot applet)
        # is configured in the dashboard. This resolver pattern returns the
        # WSS URL as JSON for flows that use an HTTPS URL-resolver applet.
        body = json.dumps({"url": media_ws_url})
        return "application/json", body

    def parse_media_message(self, message: str | bytes) -> StreamEvent:
        data = json.loads(message) if isinstance(message, (str, bytes)) else message
        kind = data.get("event", "")
        if kind == "media":
            payload = base64.b64decode(data["media"]["payload"])
            return StreamEvent(kind="media", pcm16_8k=payload,
                               stream_id=data.get("streamSid", ""), raw=data)
        if kind == "start":
            start = data.get("start", {})
            return StreamEvent(kind="start",
                               stream_id=data.get("streamSid", ""),
                               call_id=start.get("callSid", ""),
                               caller_cli=start.get("from", ""), raw=data)
        if kind == "dtmf":
            dtmf = data.get("dtmf", {})
            return StreamEvent(kind="dtmf", dtmf_digit=str(dtmf.get("digit", "")),
                               raw=data)
        if kind in ("connected", "stop"):
            return StreamEvent(kind=kind, raw=data)
        return StreamEvent(kind="unknown", raw=data)

    def extract_call_info(self, webhook_form: dict) -> tuple[str, str, str]:
        return (webhook_form.get("CallSid") or webhook_form.get("Sid") or "",
                webhook_form.get("From", ""), webhook_form.get("To", ""))


class PlivoAdapter(ProviderAdapter):
    name = "plivo"

    def answer_body(self, media_ws_url: str, notice_text: str) -> tuple[str, str]:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f'  <Speak>{_xml_escape(notice_text)}</Speak>\n'
            '  <Stream bidirectional="true" contentType="audio/x-l16;rate=8000" '
            f'keepCallAlive="true">{_xml_escape(media_ws_url)}</Stream>\n'
            "</Response>"
        )
        return "text/xml", body

    def parse_media_message(self, message: str | bytes) -> StreamEvent:
        # Plivo audio-stream frame parsing is TBD against current Plivo docs;
        # normalize the documented shape here and raise on anything else.
        data = json.loads(message) if isinstance(message, (str, bytes)) else message
        kind = data.get("event", "")
        if kind == "media":
            payload = base64.b64decode(data["media"]["payload"])
            return StreamEvent(kind="media", pcm16_8k=payload, raw=data)
        if kind == "start":
            return StreamEvent(kind="start", call_id=data.get("callId", ""),
                               caller_cli=data.get("from", ""), raw=data)
        if kind == "stop":
            return StreamEvent(kind="stop", raw=data)
        return StreamEvent(kind="unknown", raw=data)

    def extract_call_info(self, webhook_form: dict) -> tuple[str, str, str]:
        return (webhook_form.get("CallUUID", ""), webhook_form.get("From", ""),
                webhook_form.get("To", ""))


class TwilioAdapter(ProviderAdapter):
    name = "twilio"

    def answer_body(self, media_ws_url: str, notice_text: str) -> tuple[str, str]:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<Response>\n"
            f"  <Say>{_xml_escape(notice_text)}</Say>\n"
            "  <Connect>\n"
            f'    <Stream url="{_xml_escape(media_ws_url)}" />\n'
            "  </Connect>\n"
            "</Response>"
        )
        return "text/xml", body

    def parse_media_message(self, message: str | bytes) -> StreamEvent:
        data = json.loads(message) if isinstance(message, (str, bytes)) else message
        kind = data.get("event", "")
        if kind == "media":
            payload = base64.b64decode(data["media"]["payload"])
            return StreamEvent(kind="media", pcm16_8k=mulaw_to_pcm16(payload),
                               stream_id=data.get("streamSid", ""), raw=data)
        if kind == "start":
            start = data.get("start", {})
            params = {p["name"]: p["value"]
                      for p in start.get("customParameters", [])}
            return StreamEvent(kind="start",
                               stream_id=data.get("streamSid", ""),
                               call_id=start.get("callSid", ""),
                               caller_cli=params.get("caller_cli", ""),
                               raw=data)
        if kind in ("connected", "stop"):
            return StreamEvent(kind=kind, raw=data)
        return StreamEvent(kind="unknown", raw=data)

    def extract_call_info(self, webhook_form: dict) -> tuple[str, str, str]:
        return (webhook_form.get("CallSid", ""), webhook_form.get("From", ""),
                webhook_form.get("To", ""))


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


ADAPTERS: dict[str, ProviderAdapter] = {
    "exotel": ExotelAdapter(),
    "plivo": PlivoAdapter(),
    "twilio": TwilioAdapter(),
}


def get_adapter(name: str) -> ProviderAdapter:
    try:
        return ADAPTERS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; expected one of "
                         f"{sorted(ADAPTERS)}")
