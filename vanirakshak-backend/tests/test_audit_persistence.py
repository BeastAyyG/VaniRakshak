"""AuditLog JSONL persistence: append, reload, chain verify, corrupt lines."""

from __future__ import annotations

import json

from vanirakshak.privacy.audit import AuditLog


def _append(log: AuditLog, sid: str = "s1", risk: float = 10.0, tier: str = "ALLOW"):
    return log.append(
        session_id=sid,
        caller_id="c1",
        claimed_identity="modi_ji",
        risk_score=risk,
        tier=tier,
        features={"cm_score": 0.1, "snr_db": 20.0},
    )


def test_roundtrip_reloads_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(max_events=100, path=path)
    _append(log)
    _append(log, sid="s2", risk=80.0, tier="BLOCK")

    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    reloaded = AuditLog(max_events=100, path=path)
    assert len(reloaded.tail(10)) == 2
    assert reloaded.verify_chain()
    assert [e.session_id for e in reloaded.tail(10)] == ["s1", "s2"]
    assert reloaded.tail(10)[-1].tier == "BLOCK"


def test_reload_continues_chain_from_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(max_events=100, path=path)
    first = _append(log)

    reloaded = AuditLog(max_events=100, path=path)
    second = _append(reloaded, sid="s2")

    assert second.prev_hash == first.receipt_hash
    assert reloaded.verify_chain()


def test_corrupt_lines_skipped_chain_still_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(max_events=100, path=path)
    _append(log)
    with open(path, "a", encoding="utf-8") as f:
        f.write("this is not json\n")
    reloaded = AuditLog(max_events=100, path=path)
    assert len(reloaded.tail(10)) == 1
    assert reloaded.verify_chain()


def test_tampered_file_fails_chain_but_loads(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(max_events=100, path=path)
    _append(log)
    _append(log, sid="s2")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    doc = json.loads(lines[1])
    doc["risk_score"] = 0.0  # tamper without fixing the hash
    lines[1] = json.dumps(doc)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reloaded = AuditLog(max_events=100, path=path)
    assert len(reloaded.tail(10)) == 2
    assert not reloaded.verify_chain()


def test_no_path_is_memory_only(tmp_path):
    log = AuditLog(max_events=100)
    _append(log)
    assert log.verify_chain()
    assert len(log.tail(10)) == 1
    # nothing written anywhere near tmp
    assert list(tmp_path.iterdir()) == []


def test_ring_cap_still_applies_but_file_keeps_all(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(max_events=2, path=path)
    for i in range(4):
        _append(log, sid=f"s{i}")
    assert len(log.tail(10)) == 2  # hot window capped
    reloaded = AuditLog(max_events=100, path=path)
    assert len(reloaded.tail(10)) == 4  # durable record complete
    assert reloaded.verify_chain()
