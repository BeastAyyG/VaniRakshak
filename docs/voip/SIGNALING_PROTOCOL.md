# VaniRakshak Call Signaling Protocol

Phase 2 signaling server for WebRTC call setup between Android clients.
Verified live on lab-beast 2026-10-09 (17/17 handshake steps PASS).

## Endpoint

| Where | URL |
|---|---|
| lab-beast localhost | `ws://127.0.0.1:8001/ws/signal` |
| lab-beast over Tailscale | `ws://100.89.248.108:8001/ws/signal` |

Transport: WebSocket, text frames only. Every message is a JSON object with a
string `type` field. Binary frames get a `PROTOCOL` error. Server binds
`0.0.0.0:8001`. Implementation: `C:\VaniRakshak-voip\voip-signaling\server.py`
(asyncio + websockets), launched via
`C:\VaniRakshak-voip\ops\launch_signaling.cmd`, logs at
`C:\vr-out\voip-build-logs\signaling.log`.

## Registration

Client to server:

```json
{"type": "register", "user_id": "alice"}
```

Server to client:

```json
{"type": "registered", "user_id": "alice"}
```

Rules: registration is required before any other message type (else
`NOT_REGISTERED`). Registering the same `user_id` from a second socket
replaces the old one; the old socket becomes unregistered but is not
force-closed. On socket close the registration is removed, and if the user
was in a call the peer receives `{"type": "hangup", "call_id": "<uuid>"}`.

## Presence

Client to server:

```json
{"type": "presence_request"}
```

Server to client:

```json
{"type": "presence", "users": ["alice", "bob"]}
```

`users` is the sorted list of currently registered user ids.

## Call setup

Caller to server:

```json
{"type": "call", "from": "alice", "to": "bob", "call_id": "<uuid>"}
```

`from` must equal the sender's registered user id. `call_id` must be unique
(not already in use). If `to` is offline the caller gets
`{"type": "error", "code": "USER_OFFLINE", "message": "..."}`.

Server to callee:

```json
{"type": "incoming_call", "from": "alice", "call_id": "<uuid>"}
```

Callee to server (accept):

```json
{"type": "accept", "call_id": "<uuid>"}
```

Server to caller:

```json
{"type": "accepted", "call_id": "<uuid>"}
```

Callee to server (reject):

```json
{"type": "reject", "call_id": "<uuid>", "reason": "busy"}
```

`reason` is optional (defaults to `"declined"`).

Server to caller:

```json
{"type": "rejected", "call_id": "<uuid>", "reason": "busy"}
```

## Media negotiation (relayed verbatim to the peer)

Either party to server:

```json
{"type": "sdp", "call_id": "<uuid>", "kind": "offer", "sdp": {"sdp": "v=0...", "type": "offer"}}
{"type": "sdp", "call_id": "<uuid>", "kind": "answer", "sdp": {"sdp": "v=0...", "type": "answer"}}
{"type": "ice", "call_id": "<uuid>", "candidate": {"candidate": "candidate:1 ...", "sdpMid": "0"}}
```

`kind` must be `"offer"` or `"answer"`. The server forwards the message to the
other party unchanged. The `sdp` and `candidate` objects are opaque to the
server (any JSON). Unknown `call_id` gives `UNKNOWN_CALL`.

## Hangup

Either party to server:

```json
{"type": "hangup", "call_id": "<uuid>"}
```

Server to the other party:

```json
{"type": "hangup", "call_id": "<uuid>"}
```

The call record is deleted. A hangup for an unknown `call_id` gives
`UNKNOWN_CALL`.

## Call-state diagram

```
                    +------+
                    | idle |  (no call record)
                    +------+
                       |
              C->S {"type":"call"}
                       v
                    +---------+
                    | calling |  (caller view; waiting for callee)
                    +---------+
                       |
         S->callee {"type":"incoming_call"}
                       v
                    +---------+
              +---->| ringing |<----+
              |     +---------+     |
   callee hangs up / disconnects    |  callee accepts
              |                     v
              |               +--------+
              |               | active |  (sdp/ice relayed, media flows peer to peer)
              |               +--------+
              |                     |
              |        either party sends {"type":"hangup"}
              |        or a socket closes
              v                     v
                    +-------+
                    | ended |  (call record deleted; hangup sent to peer)
                    +-------+
```

Server-side state: `user_id -> websocket`; `call_id -> {from, to, state}`
where state is `"ringing"` or `"active"`. Reject moves ringing straight to
ended. A second `accept` on an already-active call is a no-op re-notify
(the caller gets `accepted` again; harmless).

## Errors

Shape:

```json
{"type": "error", "code": "USER_OFFLINE", "message": "user 'carol' is not online"}
```

| Code | Meaning |
|---|---|
| `USER_OFFLINE` | callee (or peer) is not registered / went offline |
| `UNKNOWN_CALL` | `call_id` does not match any live call |
| `NOT_REGISTERED` | any message before `register` (except `register` itself) |
| `PROTOCOL` | malformed JSON, bad field shapes, unknown message type, `from` mismatch with registered user, duplicate `call_id`, self-call, bad `sdp` kind, non-party touching a call |

## Notes for the Android workstream

* The server is signaling only; media is WebRTC peer to peer (or via a TURN
  server the app configures). The server never sees audio.
* Generate `call_id` client-side as a UUID per call attempt.
* After `accepted`, exchange SDP offer/answer then trickle ICE through the
  relayed `sdp`/`ice` messages.
* On call end (local or remote hangup, or socket drop) the app should close
  its peer connection; the server already cleans up its call record.
* Presence is pull-based (`presence_request`); there is no push on
  user join/leave in this version. Re-request presence when the call screen
  opens.
