"""VaniRakshak Layer-2 PSTN screening server (skeleton).

Provider-agnostic call screening: a CPaaS provider (Exotel, Plivo, Twilio)
answers an inbound PSTN call, streams caller audio to this server over a
WebSocket, and this server pipes the audio to the scorer WebSocket on
lab-beast, applies the verdict policy, and tells the provider what to do.

Run (no provider credentials required for import and health checks):
    uvicorn sip_screening_server.app:app --host 0.0.0.0 --port 8080
"""

__version__ = "0.1.0-phase5-skeleton"
