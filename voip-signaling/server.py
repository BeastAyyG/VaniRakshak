#!/usr/bin/env python3
r"""VaniRakshak VoIP signaling server (Phase 2).

JSON-over-WebSocket call signaling for the VaniRakshak Call feature.
Listens on 0.0.0.0:8001, endpoint /ws/signal.

Run:  C:\VaniRakshak\.venv\Scripts\python.exe server.py
Logs: C:\vr-out\voip-build-logs\signaling.log  (via launch_signaling.cmd)

Full protocol: SIGNALING_PROTOCOL.md (voip-build docs).
"""

import asyncio
import json
import time

import websockets

HOST = "0.0.0.0"
PORT = 8001
ENDPOINT = "/ws/signal"

# user_id -> websocket
users = {}
# websocket -> user_id (reverse map, for cleanup on disconnect)
conn_user = {}
# call_id -> {"from": caller_id, "to": callee_id, "state": "ringing" | "active"}
calls = {}


def log(*parts):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), *parts, flush=True)


def err(code, message):
    return {"type": "error", "code": code, "message": message}


def get_path(ws):
    req = getattr(ws, "request", None)
    if req is not None and getattr(req, "path", None):
        return req.path
    return getattr(ws, "path", ENDPOINT)


async def safe_send(ws, obj):
    try:
        await ws.send(json.dumps(obj))
        return True
    except Exception:
        return False


def peer_of(call, user):
    if user == call["from"]:
        return call["to"]
    if user == call["to"]:
        return call["from"]
    return None


async def handle_register(ws, msg):
    user_id = msg.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        await safe_send(ws, err("PROTOCOL", "register needs a non-empty user_id"))
        return
    old = users.get(user_id)
    if old is not None and old is not ws:
        # Duplicate register replaces the old socket; the old socket
        # becomes unregistered (it is NOT force-closed).
        conn_user.pop(old, None)
        log("register: replaced stale socket for", user_id)
    users[user_id] = ws
    conn_user[ws] = user_id
    log("registered", user_id)
    await safe_send(ws, {"type": "registered", "user_id": user_id})


async def handle_call(ws, user, msg):
    frm = msg.get("from")
    to = msg.get("to")
    call_id = msg.get("call_id")
    if frm != user:
        await safe_send(ws, err("PROTOCOL", "'from' must match the registered user"))
        return
    if not isinstance(to, str) or not to:
        await safe_send(ws, err("PROTOCOL", "call needs a non-empty 'to'"))
        return
    if to == user:
        await safe_send(ws, err("PROTOCOL", "cannot call yourself"))
        return
    if not isinstance(call_id, str) or not call_id:
        await safe_send(ws, err("PROTOCOL", "call needs a non-empty call_id"))
        return
    if call_id in calls:
        await safe_send(ws, err("PROTOCOL", "call_id already in use"))
        return
    callee_ws = users.get(to)
    if callee_ws is None:
        await safe_send(ws, err("USER_OFFLINE", "user '%s' is not online" % to))
        return
    calls[call_id] = {"from": user, "to": to, "state": "ringing"}
    log("call", call_id, user, "->", to)
    ok = await safe_send(callee_ws, {"type": "incoming_call", "from": user, "call_id": call_id})
    if not ok:
        del calls[call_id]
        await safe_send(ws, err("USER_OFFLINE", "user '%s' is not online" % to))


async def handle_accept(ws, user, msg):
    call_id = msg.get("call_id")
    call = calls.get(call_id) if isinstance(call_id, str) else None
    if call is None:
        await safe_send(ws, err("UNKNOWN_CALL", "no such call"))
        return
    if call["to"] != user:
        await safe_send(ws, err("PROTOCOL", "only the callee can accept"))
        return
    call["state"] = "active"
    log("accept", call_id)
    caller_ws = users.get(call["from"])
    if caller_ws is None or not await safe_send(caller_ws, {"type": "accepted", "call_id": call_id}):
        await safe_send(ws, err("USER_OFFLINE", "caller went offline"))


async def handle_reject(ws, user, msg):
    call_id = msg.get("call_id")
    call = calls.get(call_id) if isinstance(call_id, str) else None
    if call is None:
        await safe_send(ws, err("UNKNOWN_CALL", "no such call"))
        return
    if call["to"] != user:
        await safe_send(ws, err("PROTOCOL", "only the callee can reject"))
        return
    reason = msg.get("reason") or "declined"
    del calls[call_id]
    log("reject", call_id, reason)
    caller_ws = users.get(call["from"])
    if caller_ws is not None:
        await safe_send(caller_ws, {"type": "rejected", "call_id": call_id, "reason": reason})


async def handle_relay(ws, user, msg):
    # sdp and ice: relayed to the peer unchanged.
    call_id = msg.get("call_id")
    call = calls.get(call_id) if isinstance(call_id, str) else None
    if call is None:
        await safe_send(ws, err("UNKNOWN_CALL", "no such call"))
        return
    peer_id = peer_of(call, user)
    if peer_id is None:
        await safe_send(ws, err("PROTOCOL", "you are not a party to this call"))
        return
    if msg["type"] == "sdp" and msg.get("kind") not in ("offer", "answer"):
        await safe_send(ws, err("PROTOCOL", "sdp kind must be 'offer' or 'answer'"))
        return
    peer_ws = users.get(peer_id)
    if peer_ws is None or not await safe_send(peer_ws, msg):
        del calls[call_id]
        await safe_send(ws, err("USER_OFFLINE", "peer went offline"))


async def handle_hangup(ws, user, msg):
    call_id = msg.get("call_id")
    call = calls.get(call_id) if isinstance(call_id, str) else None
    if call is None:
        await safe_send(ws, err("UNKNOWN_CALL", "no such call"))
        return
    peer_id = peer_of(call, user)
    if peer_id is None:
        await safe_send(ws, err("PROTOCOL", "you are not a party to this call"))
        return
    del calls[call_id]
    log("hangup", call_id)
    peer_ws = users.get(peer_id)
    if peer_ws is not None:
        await safe_send(peer_ws, {"type": "hangup", "call_id": call_id})


async def handle_message(ws, raw):
    try:
        msg = json.loads(raw)
    except Exception:
        await safe_send(ws, err("PROTOCOL", "invalid JSON"))
        return
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        await safe_send(ws, err("PROTOCOL", "message must be a JSON object with a 'type'"))
        return
    mtype = msg["type"]
    if mtype == "register":
        await handle_register(ws, msg)
        return
    user = conn_user.get(ws)
    if user is None:
        await safe_send(ws, err("NOT_REGISTERED", "register first"))
        return
    if mtype == "presence_request":
        await safe_send(ws, {"type": "presence", "users": sorted(users.keys())})
    elif mtype == "call":
        await handle_call(ws, user, msg)
    elif mtype == "accept":
        await handle_accept(ws, user, msg)
    elif mtype == "reject":
        await handle_reject(ws, user, msg)
    elif mtype in ("sdp", "ice"):
        await handle_relay(ws, user, msg)
    elif mtype == "hangup":
        await handle_hangup(ws, user, msg)
    else:
        await safe_send(ws, err("PROTOCOL", "unknown message type '%s'" % mtype))


async def cleanup(ws):
    user = conn_user.pop(ws, None)
    if user is None:
        return
    if users.get(user) is ws:
        del users[user]
        log("unregistered", user)
    # If the user was in any call, notify the peer with a hangup.
    for call_id, call in list(calls.items()):
        if user in (call["from"], call["to"]):
            peer_id = call["to"] if user == call["from"] else call["from"]
            del calls[call_id]
            peer_ws = users.get(peer_id)
            if peer_ws is not None:
                await safe_send(peer_ws, {"type": "hangup", "call_id": call_id})
            log("hangup-on-disconnect", call_id)


async def handler(ws, *args):
    if get_path(ws) != ENDPOINT:
        await ws.close(code=1008, reason="unknown endpoint")
        return
    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                await safe_send(ws, err("PROTOCOL", "binary frames not supported"))
                continue
            await handle_message(ws, raw)
    except websockets.ConnectionClosed:
        pass
    except Exception as exc:  # never kill the server on one bad connection
        log("handler error:", repr(exc))
    finally:
        await cleanup(ws)


async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=1 << 20):
        log("signaling server listening on %s:%d%s" % (HOST, PORT, ENDPOINT))
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
