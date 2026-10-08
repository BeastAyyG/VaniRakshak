#!/usr/bin/env python3
"""
Minimal RFC 5766 TURN server (UDP only, long-term credentials) for lab testing.

Supports: Binding, Allocate, Refresh, CreatePermission, Send/Data indications,
ChannelBind/ChannelData. Single-file, stdlib only (asyncio).

Usage:
  python turn_server.py --port 3478 --external-ip 10.0.2.2
Credentials: username "vr", password "vanirakshak-turn", realm "vanirakshak".
Logs to stdout (redirect to a file when launched via WMI).
"""

import argparse
import asyncio
import hashlib
import hmac
import secrets
import socket
import struct
import time
import zlib

MAGIC = 0x2112A442
REALM = "vanirakshak"
USERNAME = "vr"
PASSWORD=REPLACE_WITH_YOUR_OWN_SECRET

# Message types
T_BINDING_REQ = 0x0001
T_BINDING_OK = 0x0101
T_ALLOCATE_REQ = 0x0003
T_ALLOCATE_OK = 0x0103
T_ALLOCATE_ERR = 0x0113
T_REFRESH_REQ = 0x0004
T_REFRESH_OK = 0x0104
T_CREATEPERM_REQ = 0x0008
T_CREATEPERM_OK = 0x0108
T_SEND_IND = 0x0016
T_DATA_IND = 0x0017
T_CHANNELBIND_REQ = 0x0009
T_CHANNELBIND_OK = 0x0109
T_GENERIC_ERR = 0x0111

# Attribute types
A_XOR_MAPPED = 0x0020
A_XOR_RELAYED = 0x0016
A_XOR_PEER = 0x0012
A_USERNAME = 0x0006
A_MSG_INTEGRITY = 0x0008
A_ERROR_CODE = 0x0009
A_REALM = 0x0014
A_NONCE = 0x0015
A_REQ_TRANSPORT = 0x0019
A_LIFETIME = 0x000D
A_CHANNEL_NUM = 0x000C
A_DATA = 0x0013
A_FINGERPRINT = 0x8028

KEY = hashlib.md5(f"{USERNAME}:{REALM}:{PASSWORD}".encode()).digest()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def xor_addr(ip, port):
    ipb = socket.inet_aton(ip)
    xport = port ^ (MAGIC >> 16)
    xip = bytes(a ^ b for a, b in zip(ipb, struct.pack("!I", MAGIC)))
    return b"\x00\x01" + struct.pack("!H", xport) + xip


def parse_xor_addr(value):
    family = value[1]
    xport = struct.unpack("!H", value[2:4])[0]
    port = xport ^ (MAGIC >> 16)
    if family == 0x01:
        xip = value[4:8]
        ipb = bytes(a ^ b for a, b in zip(xip, struct.pack("!I", MAGIC)))
        return socket.inet_ntoa(ipb), port
    raise ValueError("only IPv4 supported")


def parse_attrs(data):
    """Return list of (type, value, value_offset)."""
    attrs = []
    pos = 20
    while pos + 4 <= len(data):
        atype, alen = struct.unpack("!HH", data[pos:pos + 4])
        val = data[pos + 4:pos + 4 + alen]
        attrs.append((atype, val, pos + 4))
        pos += 4 + alen + (-alen % 4)
    return attrs


def verify_mi(data, attrs):
    mi = next(((t, v, o) for (t, v, o) in attrs if t == A_MSG_INTEGRITY), None)
    if mi is None:
        return False
    _, _, voff = mi
    username = next((v for (t, v, o) in attrs if t == A_USERNAME), None)
    if username != USERNAME.encode():
        return False
    trunc_end = voff + 20
    trunc = bytearray(data[:trunc_end])
    struct.pack_into("!H", trunc, 2, trunc_end - 20)
    trunc[voff:voff + 20] = b"\x00" * 20
    calc = hmac.new(KEY, bytes(trunc), hashlib.sha1).digest()
    return hmac.compare_digest(calc, data[voff:voff + 20])


def build_message(msg_type, txn_id, attrs, with_mi=True):
    body = b""
    for atype, aval in attrs:
        body += struct.pack("!HH", atype, len(aval)) + aval + b"\x00" * (-len(aval) % 4)
    mi_attr = b""
    if with_mi:
        mi_attr = struct.pack("!HH", A_MSG_INTEGRITY, 20) + b"\x00" * 20
    fp_attr = struct.pack("!HH", A_FINGERPRINT, 4) + b"\x00" * 4
    # RFC 5389: the MI HMAC is computed with the header length covering the
    # message only up to and including the MI attribute (the fingerprint,
    # which follows MI, is not covered by MI).
    mi_len = len(body) + len(mi_attr)
    header = struct.pack("!HHI", msg_type, mi_len, MAGIC) + txn_id
    msg = header + body + mi_attr
    if with_mi:
        sig = hmac.new(KEY, msg, hashlib.sha1).digest()
        msg = msg[:-20] + sig
    # Final header length includes the fingerprint attribute appended next.
    full_len = mi_len + len(fp_attr)
    msg = struct.pack("!HHI", msg_type, full_len, MAGIC) + txn_id + msg[20:]
    fp = zlib.crc32(msg) ^ 0x5354554E
    msg += struct.pack("!HHI", A_FINGERPRINT, 4, fp)
    return msg


def error_code_attr(code, reason):
    # RFC 5389 15.6: 2 zero octets, then class (hundreds digit), then number.
    return struct.pack("!HBB", 0, code // 100, code % 100) + reason.encode()


class Allocation:
    def __init__(self, client_addr, relay_sock, external_ip):
        self.client_addr = client_addr
        self.relay_sock = relay_sock
        self.external_ip = external_ip
        self.relay_port = relay_sock.getsockname()[1]
        self.permissions = {}  # (ip,port) -> expiry
        self.channels = {}  # channel_no -> (ip,port)
        self.peer_channel = {}  # (ip,port) -> channel_no
        self.expiry = time.time() + 600


class TurnServerProtocol(asyncio.DatagramProtocol):
    def __init__(self, external_ip):
        self.external_ip = external_ip
        self.transport = None
        self.allocations = {}  # client_addr -> Allocation
        self.loop = asyncio.get_event_loop()

    def connection_made(self, transport):
        self.transport = transport
        log(f"TURN listening on UDP 0.0.0.0, external-ip={self.external_ip}")

    def datagram_received(self, data, addr):
        try:
            if len(data) < 4:
                return
            if data[0] & 0xC0 == 0x40:
                self.handle_channel_data(data, addr)
            elif data[0] & 0xC0 == 0x00:
                self.handle_stun(data, addr)
        except Exception as e:
            log(f"datagram error from {addr}: {e}")

    # ---- ChannelData (0x4000-0x7FFF) ----
    def handle_channel_data(self, data, addr):
        if len(data) < 4:
            return
        ch_no, dlen = struct.unpack("!HH", data[:4])
        payload = data[4:4 + dlen]
        alloc = self.allocations.get(addr)
        if alloc is None:
            return
        peer = alloc.channels.get(ch_no)
        if peer is None:
            return
        alloc.permissions[peer] = time.time() + 300
        alloc.relay_sock.sendto(payload, peer)

    # ---- STUN ----
    def handle_stun(self, data, addr):
        if len(data) < 20:
            return
        msg_type, msg_len, magic = struct.unpack("!HHI", data[:8])
        if magic != MAGIC:
            return
        txn_id = data[8:20]
        attrs = parse_attrs(data)
        ad = {t: v for (t, v, o) in attrs}

        if msg_type == T_BINDING_REQ:
            resp = build_message(T_BINDING_OK, txn_id,
                                 [(A_XOR_MAPPED, xor_addr(addr[0], addr[1]))], with_mi=False)
            self.transport.sendto(resp, addr)
            return

        if msg_type == T_SEND_IND:
            alloc = self.allocations.get(addr)
            if alloc is None or A_XOR_PEER not in ad or A_DATA not in ad:
                return
            peer = parse_xor_addr(ad[A_XOR_PEER])
            if peer not in alloc.permissions or alloc.permissions[peer] < time.time():
                return
            alloc.relay_sock.sendto(ad[A_DATA], peer)
            return

        # Requests below need an allocation except Allocate itself.
        if msg_type == T_ALLOCATE_REQ:
            self.handle_allocate(data, addr, txn_id, attrs, ad)
            return

        alloc = self.allocations.get(addr)
        if alloc is None:
            return
        if not verify_mi(data, attrs):
            self.send_error(addr, txn_id, T_GENERIC_ERR, 401, "Unauthorized")
            return

        if msg_type == T_REFRESH_REQ:
            lifetime = struct.unpack("!I", ad.get(A_LIFETIME, struct.pack("!I", 600)))[0]
            if lifetime == 0:
                self.drop_allocation(addr)
            else:
                alloc.expiry = time.time() + min(lifetime, 3600)
            resp = build_message(T_REFRESH_OK, txn_id, [(A_LIFETIME, struct.pack("!I", 600))])
            self.transport.sendto(resp, addr)
        elif msg_type == T_CREATEPERM_REQ:
            peer = parse_xor_addr(ad[A_XOR_PEER])
            alloc.permissions[peer] = time.time() + 300
            log(f"permission {addr} -> {peer}")
            resp = build_message(T_CREATEPERM_OK, txn_id, [])
            self.transport.sendto(resp, addr)
        elif msg_type == T_CHANNELBIND_REQ:
            ch_no = struct.unpack("!H", ad[A_CHANNEL_NUM][:2])[0]
            peer = parse_xor_addr(ad[A_XOR_PEER])
            alloc.channels[ch_no] = peer
            alloc.peer_channel[peer] = ch_no
            alloc.permissions[peer] = time.time() + 600
            log(f"channelbind {addr} ch={hex(ch_no)} -> {peer}")
            resp = build_message(T_CHANNELBIND_OK, txn_id, [])
            self.transport.sendto(resp, addr)

    def handle_allocate(self, data, addr, txn_id, attrs, ad):
        if not verify_mi(data, attrs):
            nonce = secrets.token_hex(8)
            resp = build_message(
                T_ALLOCATE_ERR, txn_id,
                [(A_ERROR_CODE, error_code_attr(401, "Unauthorized")),
                 (A_REALM, REALM.encode()),
                 (A_NONCE, nonce.encode())],
                with_mi=False)
            self.transport.sendto(resp, addr)
            return
        req_transport = ad.get(A_REQ_TRANSPORT, b"\x11\x00\x00\x00")
        if req_transport[0] != 17:
            self.send_error(addr, txn_id, T_ALLOCATE_ERR, 442, "Unsupported Transport Protocol")
            return
        # Drop any previous allocation for this client.
        self.drop_allocation(addr)
        relay = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        relay.bind(("0.0.0.0", 0))
        relay.setblocking(False)
        alloc = Allocation(addr, relay, self.external_ip)
        self.allocations[addr] = alloc
        self.loop.add_reader(relay.fileno(), self.on_relay_data, alloc)
        log(f"allocated {addr} -> relay {self.external_ip}:{alloc.relay_port}")
        resp = build_message(
            T_ALLOCATE_OK, txn_id,
            [(A_XOR_RELAYED, xor_addr(self.external_ip, alloc.relay_port)),
             (A_XOR_MAPPED, xor_addr(addr[0], addr[1])),
             (A_LIFETIME, struct.pack("!I", 600))])
        self.transport.sendto(resp, addr)

    def on_relay_data(self, alloc):
        try:
            payload, peer = alloc.relay_sock.recvfrom(65535)
        except BlockingIOError:
            return
        if peer not in alloc.permissions or alloc.permissions[peer] < time.time():
            return
        ch_no = alloc.peer_channel.get(peer)
        if ch_no is not None:
            msg = struct.pack("!HH", ch_no, len(payload)) + payload
            self.transport.sendto(msg, alloc.client_addr)
        else:
            msg_type = T_DATA_IND
            attrs = [(A_XOR_PEER, xor_addr(peer[0], peer[1])), (A_DATA, payload)]
            # Build indication without MI.
            body = b""
            for atype, aval in attrs:
                body += struct.pack("!HH", atype, len(aval)) + aval + b"\x00" * (-len(aval) % 4)
            header = struct.pack("!HHI", msg_type, len(body), MAGIC) + secrets.token_bytes(12)
            self.transport.sendto(header + body, alloc.client_addr)

    def drop_allocation(self, addr):
        alloc = self.allocations.pop(addr, None)
        if alloc is not None:
            self.loop.remove_reader(alloc.relay_sock.fileno())
            alloc.relay_sock.close()
            log(f"allocation dropped for {addr}")

    def send_error(self, addr, txn_id, msg_type, code, reason):
        resp = build_message(msg_type, txn_id,
                             [(A_ERROR_CODE, error_code_attr(code, reason))], with_mi=False)
        self.transport.sendto(resp, addr)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3478)
    ap.add_argument("--external-ip", default="127.0.0.1")
    args = ap.parse_args()
    loop = asyncio.get_event_loop()
    await loop.create_datagram_endpoint(
        lambda: TurnServerProtocol(args.external_ip),
        local_addr=("0.0.0.0", args.port))
    log(f"TURN server up on UDP port {args.port}")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
