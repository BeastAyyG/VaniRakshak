#!/usr/bin/env python3
"""End-to-end handshake test for the VaniRakshak VoIP signaling server.

Drives two concurrent websocket clients (alice, bob) through the full
call flow plus negative cases. Every step is asserted; prints PASS/FAIL
per step and exits non-zero on any failure.

Usage: python test_signaling.py [ws_url]
Default URL: ws://127.0.0.1:8001/ws/signal
"""

import asyncio
import json
import sys
import uuid

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8001/ws/signal"
TIMEOUT = 5.0

failures = []


def check(name, cond, detail=""):
    ok = bool(cond)
    suffix = " | detail: %s" % detail if (detail and not ok) else ""
    print(("PASS" if ok else "FAIL"), "-", name + suffix, flush=True)
    if not ok:
        failures.append(name)


async def recv(ws):
    raw = await asyncio.wait_for(asyncio.ensure_future(ws.recv()), TIMEOUT)
    assert isinstance(raw, str), "expected text frame, got %r" % type(raw)
    return json.loads(raw)


async def main():
    alice = await websockets.connect(URL)
    bob = await websockets.connect(URL)
    stranger = await websockets.connect(URL)

    # 1. register alice
    await alice.send(json.dumps({"type": "register", "user_id": "alice"}))
    check("register alice",
          await recv(alice) == {"type": "registered", "user_id": "alice"})

    # 2. register bob
    await bob.send(json.dumps({"type": "register", "user_id": "bob"}))
    check("register bob",
          await recv(bob) == {"type": "registered", "user_id": "bob"})

    # 3. presence
    await alice.send(json.dumps({"type": "presence_request"}))
    m = await recv(alice)
    check("presence lists alice+bob",
          m.get("type") == "presence" and sorted(m.get("users", [])) == ["alice", "bob"], m)

    # 4. alice calls bob
    call_id = "test-" + uuid.uuid4().hex[:8]
    await alice.send(json.dumps({"type": "call", "from": "alice", "to": "bob",
                                 "call_id": call_id}))
    m = await recv(bob)
    check("bob gets incoming_call",
          m == {"type": "incoming_call", "from": "alice", "call_id": call_id}, m)

    # 5. bob accepts
    await bob.send(json.dumps({"type": "accept", "call_id": call_id}))
    m = await recv(alice)
    check("alice gets accepted",
          m == {"type": "accepted", "call_id": call_id}, m)

    # 6. SDP offer alice -> bob (relayed verbatim)
    offer = {"type": "sdp", "call_id": call_id, "kind": "offer",
             "sdp": {"sdp": "v=0 fake-offer", "type": "offer"}}
    await alice.send(json.dumps(offer))
    check("bob gets sdp offer", await recv(bob) == offer)

    # 7. SDP answer bob -> alice (relayed verbatim)
    answer = {"type": "sdp", "call_id": call_id, "kind": "answer",
              "sdp": {"sdp": "v=0 fake-answer", "type": "answer"}}
    await bob.send(json.dumps(answer))
    check("alice gets sdp answer", await recv(alice) == answer)

    # 8. ICE alice -> bob
    ice_a = {"type": "ice", "call_id": call_id,
             "candidate": {"candidate": "candidate:1", "sdpMid": "0"}}
    await alice.send(json.dumps(ice_a))
    check("bob gets ice", await recv(bob) == ice_a)

    # 9. ICE bob -> alice
    ice_b = {"type": "ice", "call_id": call_id,
             "candidate": {"candidate": "candidate:2", "sdpMid": "0"}}
    await bob.send(json.dumps(ice_b))
    check("alice gets ice", await recv(alice) == ice_b)

    # 10. alice hangs up
    await alice.send(json.dumps({"type": "hangup", "call_id": call_id}))
    m = await recv(bob)
    check("bob gets hangup",
          m == {"type": "hangup", "call_id": call_id}, m)

    # 11. negative: call to an offline user
    await alice.send(json.dumps({"type": "call", "from": "alice", "to": "carol",
                                 "call_id": "x-offline"}))
    m = await recv(alice)
    check("call to offline user -> USER_OFFLINE",
          m.get("type") == "error" and m.get("code") == "USER_OFFLINE", m)

    # 12. negative: sdp for an unknown call_id
    await alice.send(json.dumps({"type": "sdp", "call_id": "nope",
                                 "kind": "offer", "sdp": {}}))
    m = await recv(alice)
    check("sdp unknown call -> UNKNOWN_CALL",
          m.get("type") == "error" and m.get("code") == "UNKNOWN_CALL", m)

    # 13. negative: unregistered client
    await stranger.send(json.dumps({"type": "presence_request"}))
    m = await recv(stranger)
    check("unregistered presence -> NOT_REGISTERED",
          m.get("type") == "error" and m.get("code") == "NOT_REGISTERED", m)

    # 14. duplicate register replaces the old socket
    alice2 = await websockets.connect(URL)
    await alice2.send(json.dumps({"type": "register", "user_id": "alice"}))
    check("duplicate register acked",
          await recv(alice2) == {"type": "registered", "user_id": "alice"})
    await alice.send(json.dumps({"type": "presence_request"}))
    m = await recv(alice)
    check("old socket is unregistered after replace",
          m.get("type") == "error" and m.get("code") == "NOT_REGISTERED", m)
    await alice2.close()

    # 15. disconnect during a call notifies the peer with hangup
    dave = await websockets.connect(URL)
    erin = await websockets.connect(URL)
    await dave.send(json.dumps({"type": "register", "user_id": "dave"}))
    await recv(dave)
    await erin.send(json.dumps({"type": "register", "user_id": "erin"}))
    await recv(erin)
    cid2 = "test-" + uuid.uuid4().hex[:8]
    await dave.send(json.dumps({"type": "call", "from": "dave", "to": "erin",
                                "call_id": cid2}))
    m = await recv(erin)
    check("second call rings", m.get("type") == "incoming_call", m)
    await dave.close()  # caller vanishes mid-ring
    m = await recv(erin)
    check("peer gets hangup on disconnect",
          m == {"type": "hangup", "call_id": cid2}, m)
    await erin.close()

    await alice.close()
    await bob.close()
    await stranger.close()

    print(flush=True)
    if failures:
        print("FAILED: %d step(s): %s" % (len(failures), ", ".join(failures)))
        return 1
    print("ALL STEPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
