"""Pause and resume the from-scratch CM experiment without touching its results.

Why this is safe, and what is deliberately *not* implemented
===========================================================

There are exactly two ways to pause this pipeline that cannot change a single
number, and this script only implements those two:

1. **Suspend the live process** (``NtSuspendProcess`` on Windows, ``SIGSTOP`` on
   POSIX). Every thread stops where it is. No gradient step is discarded, no
   optimizer or LR-scheduler state is rebuilt, no RNG stream is re-seeded, no
   epoch is re-shuffled. On resume the process continues from the exact
   instruction it stopped at, so the trajectory -- and therefore the checkpoint
   and every metric derived from it -- is bit-for-bit what it would have been
   had you never paused. The cost is elapsed wall time, nothing else.

2. **Hold between runs.** S42/B42/S43/B43 are four independent processes with
   their own seeds and their own checkpoints; nothing is carried between them.
   Inserting an arbitrary delay before the next one starts is therefore also a
   no-op numerically. This is the ``handoff/PAUSE`` gate, which the orchestrator
   honours when launched with ``--respect-pause``.

What is *not* implemented, on purpose:

* **Checkpoint-and-restart mid-run.** Resuming from a checkpoint loses the
  optimizer moments, the LR schedule position, the remaining epochs of that
  run's RNG stream, and the exact sample ordering. That produces a different
  model, which would silently invalidate the pre-registered comparison in
  docs/10. If a run must be stopped, the honest thing is to stop it and report
  it, not to quietly produce a half-paused artefact.
* **Reducing epochs, batch size, or dev-validation work to "save time".** That
  is a different recipe, not a paused one.

Usage:
    python scripts/experiment_control.py status
    python scripts/experiment_control.py pause --note "freeing the GPU for 2 h"
    python scripts/experiment_control.py resume
    python scripts/experiment_control.py --self-test     # prove it on a dummy

The self-test spawns a throwaway process, suspends it, and asserts that it
neither advances while suspended nor loses a single iteration across the pause.
Run it once on the training box before trusting ``pause`` on a real run.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_state as xs  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"

# Targets we are willing to freeze. A process must look like one of these to be
# touched without --force; this is the guard that stops a typo in a PID from
# freezing something unrelated.
TARGET_MARKERS = ("train_cm.py",)


@dataclass
class Target:
    pid: int
    ppid: Optional[int] = None
    cmdline: str = ""

    @property
    def short(self) -> str:
        cmd = self.cmdline.strip()
        return cmd if len(cmd) <= 110 else cmd[:107] + "..."


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _ps_query_windows() -> List[Target]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: List[Target] = []
    for row in data:
        try:
            out.append(Target(
                pid=int(row["ProcessId"]),
                ppid=int(row["ParentProcessId"]) if row.get("ParentProcessId") else None,
                cmdline=str(row.get("CommandLine") or ""),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _ps_query_posix() -> List[Target]:
    try:
        proc = subprocess.run(["ps", "-eo", "pid=,ppid=,args="],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    out: List[Target] = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            out.append(Target(pid=int(parts[0]), ppid=int(parts[1]), cmdline=parts[2]))
        except ValueError:
            continue
    return out


def all_processes() -> List[Target]:
    return _ps_query_windows() if IS_WINDOWS else _ps_query_posix()


def process_table() -> Dict[int, Target]:
    me = os.getpid()
    return {t.pid: t for t in all_processes() if t.pid not in (0, 4, me)}


def subtree_pids(roots: Iterable[int], table: Dict[int, Target]) -> List[int]:
    """Roots plus every descendant, breadth-first.

    This is required, not cosmetic. On the training box the process whose
    command line names ``train_cm.py`` is a uv venv launcher stub -- one idle
    thread, 0% CPU -- and the interpreter actually holding the CUDA context is
    its child. Suspending only the matched process would report success and
    freeze nothing at all.
    """
    kids: Dict[int, List[int]] = {}
    for t in table.values():
        if t.ppid is not None:
            kids.setdefault(t.ppid, []).append(t.pid)

    out: List[int] = []
    seen = set()
    queue = list(dict.fromkeys(roots))
    while queue:
        pid = queue.pop(0)
        if pid in seen or pid not in table:
            continue
        seen.add(pid)
        out.append(pid)
        queue.extend(kids.get(pid, []))
    return out


def find_roots() -> List[Target]:
    table = process_table()
    return sorted(
        (t for t in table.values() if any(m in t.cmdline for m in TARGET_MARKERS)),
        key=lambda t: t.pid,
    )


def find_targets() -> List[Target]:
    """The full set of processes that must be frozen to stop the run."""
    table = process_table()
    roots = [t.pid for t in table.values()
             if any(m in t.cmdline for m in TARGET_MARKERS)]
    return [table[p] for p in subtree_pids(roots, table)]


def process_alive(pid: int) -> bool:
    if IS_WINDOWS:
        return any(t.pid == pid for t in all_processes())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# CPU accounting -- used to *verify* a pause actually took effect
# ---------------------------------------------------------------------------


def cpu_seconds(pids: Sequence[int]) -> Dict[int, float]:
    """Cumulative CPU time per pid. The check that a suspend really bit."""
    want = set(pids)
    if IS_WINDOWS:
        script = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,UserModeTime,KernelModeTime | "
            "ConvertTo-Json -Compress"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(proc.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return {}
        if isinstance(data, dict):
            data = [data]
        out: Dict[int, float] = {}
        for row in data:
            try:
                pid = int(row["ProcessId"])
            except (KeyError, TypeError, ValueError):
                continue
            if pid in want:
                out[pid] = (int(row.get("UserModeTime") or 0)
                            + int(row.get("KernelModeTime") or 0)) / 1e7
        return out

    out = {}
    ticks = os.sysconf("SC_CLK_TCK")
    for pid in want:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
            out[pid] = (int(fields[13]) + int(fields[14])) / ticks
        except (OSError, ValueError, IndexError):
            continue
    return out


def still_running_hot(pids: Sequence[int], window: float = 2.0,
                      threshold: float = 0.4) -> Dict[int, float]:
    """Pids that kept consuming CPU across ``window`` seconds.

    Called after a suspend: if anything is still burning CPU, the pause did not
    cover it, and the caller must not claim the run is frozen.
    """
    before = cpu_seconds(pids)
    time.sleep(window)
    after = cpu_seconds(pids)
    hot: Dict[int, float] = {}
    for pid in pids:
        if pid in before and pid in after:
            delta = after[pid] - before[pid]
            if delta > threshold:
                hot[pid] = delta
    return hot


# ---------------------------------------------------------------------------
# Suspend / resume
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    _PROCESS_SUSPEND_RESUME = 0x0800
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")
    _kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    _kernel32.OpenProcess.restype = ctypes.c_void_p
    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _ntdll.NtSuspendProcess.argtypes = [ctypes.c_void_p]
    _ntdll.NtSuspendProcess.restype = ctypes.c_long
    _ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
    _ntdll.NtResumeProcess.restype = ctypes.c_long


def _win_signal(pid: int, suspend: bool) -> None:
    """NtSuspendProcess / NtResumeProcess -- what psutil.suspend() does."""
    handle = _kernel32.OpenProcess(_PROCESS_SUSPEND_RESUME, 0, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed")
    try:
        fn = _ntdll.NtSuspendProcess if suspend else _ntdll.NtResumeProcess
        status = fn(handle)
        if status != 0:
            # 0xC0000005/0xC0000022 => access denied; 0xC000000B => bad PID.
            raise OSError(f"Nt{'Suspend' if suspend else 'Resume'}Process({pid}) "
                          f"returned NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
    finally:
        _kernel32.CloseHandle(handle)


def suspend(pid: int) -> None:
    if IS_WINDOWS:
        _win_signal(pid, True)
    else:
        os.kill(pid, 19)  # SIGSTOP


def resume(pid: int) -> None:
    if IS_WINDOWS:
        _win_signal(pid, False)
    else:
        os.kill(pid, 18)  # SIGCONT


# ---------------------------------------------------------------------------
# Gate + control state
# ---------------------------------------------------------------------------


def _read_control(path: Path = xs.CONTROL_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_control(payload: dict, path: Path = xs.CONTROL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def set_gate(note: str) -> None:
    xs.GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    xs.GATE_PATH.write_text(
        json.dumps({
            "requested_at": xs._now().isoformat(timespec="seconds"),
            "note": note,
            "honoured_by": "scripts/run_from_scratch_experiment.py --respect-pause",
            "clear_with": "python scripts/experiment_control.py resume",
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def clear_gate() -> bool:
    try:
        xs.GATE_PATH.unlink()
        return True
    except FileNotFoundError:
        return False


def gate_present() -> bool:
    return xs.GATE_PATH.exists()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_status(as_json: bool = False) -> int:
    control = _read_control()
    table = process_table()
    roots = find_roots()
    root_pids = {t.pid for t in roots}
    tree = [table[p] for p in subtree_pids([t.pid for t in roots], table)]
    live = [t for t in tree if process_alive(t.pid)]

    if as_json:
        print(json.dumps({
            "paused": gate_present(),
            "gate_path": str(xs.GATE_PATH),
            "control": control,
            "training_tree": [
                {**asdict(t), "role": "root" if t.pid in root_pids else "child"}
                for t in live
            ],
        }, indent=2))
        return 0

    prog = xs.collect()
    running = next((r for r in prog.runs if r.state == "running"), None)
    print(f"gate        : {'PAUSED' if gate_present() else 'open'}  ({xs.GATE_PATH})")
    if gate_present():
        print(f"  requested : {control.get('since', 'unknown')}"
              f"  note: {control.get('note', '')!r}")
    print(f"experiment  : {prog.fraction * 100:.1f}% complete, "
          f"eta {prog.eta:%Y-%m-%d %H:%M}")
    print(f"current run : {running.label if running else '(none)'}")
    if live:
        print(f"training tree ({len(live)} process(es)):")
        for t in live:
            role = "root " if t.pid in root_pids else "child"
            print(f"  {role} pid {t.pid:<7} ppid {str(t.ppid):<7} {t.short[:88]}")
        print("  note: the root is a uv venv launcher stub on this box; the child "
              "is the interpreter that holds the CUDA context. Pause freezes both.")
    else:
        print("training tree: none found")
    suspended = control.get("suspended_pids") or []
    if suspended:
        print(f"suspended pids: {suspended}")
    return 0


def cmd_pause(note: str, dry_run: bool, force: bool) -> int:
    table = process_table()
    roots = find_roots()
    if force:
        roots = [t for t in table.values()]
    root_pids = [t.pid for t in roots]
    pids = [p for p in subtree_pids(root_pids, table) if p in table]
    targets = [table[p] for p in pids if process_alive(p)]

    prog = xs.collect()
    running = next((r for r in prog.runs if r.state == "running"), None)

    print(f"gate: {'already set' if gate_present() else 'will be set'} ({xs.GATE_PATH})")
    if running:
        print(f"run in flight: {running.label} "
              f"(epoch {running.current_epoch}/{running.total_epochs})")
    if not targets:
        print("no live training process found; setting the gate only, so the next "
              "run holds before it starts.")
    for t in targets:
        role = "root" if t.pid in root_pids else "child"
        print(f"  would suspend pid {t.pid:<7} [{role}] ppid {t.ppid}  {t.short[:80]}")

    if dry_run:
        print("\n--dry-run: nothing changed.")
        return 0

    # Parents first: a suspended parent cannot spawn a new child out from under
    # the pause, so the frozen set cannot grow after we measure it.
    suspended: List[int] = []
    failures: List[str] = []
    for t in sorted(targets, key=lambda t: (t.pid not in root_pids, t.pid)):
        try:
            suspend(t.pid)
            suspended.append(t.pid)
            print(f"  suspended pid {t.pid}")
        except OSError as exc:
            failures.append(f"pid {t.pid}: {exc}")
            print(f"  FAILED to suspend pid {t.pid}: {exc}", file=sys.stderr)

    # Do not report a pause we cannot demonstrate.
    hot = still_running_hot(suspended) if suspended else {}
    for pid, delta in hot.items():
        failures.append(f"pid {pid} still burning {delta:.2f}s CPU after suspend")
        print(f"  VERIFY FAILED: pid {pid} consumed {delta:.2f}s CPU while suspended",
              file=sys.stderr)
    if suspended and not hot:
        print(f"  verified: {len(suspended)} process(es) consuming no CPU")

    set_gate(note)
    _write_control({
        "paused": True,
        "since": xs._now().isoformat(timespec="seconds"),
        "note": note,
        "suspended_pids": suspended,
        "suspended_targets": [asdict(t) for t in targets if t.pid in suspended],
        "failures": failures,
    })

    print(f"\npaused. {len(suspended)} process(es) suspended; gate set.")
    if failures:
        print("some targets could not be frozen -- see above.", file=sys.stderr)
    print("The training process is frozen exactly where it is: no step is lost, "
          "no state is rebuilt. Resume with `experiment_control.py resume`.")
    return 0 if not failures else 1


def cmd_resume(dry_run: bool = False) -> int:
    control = _read_control()
    pids = [int(p) for p in (control.get("suspended_pids") or [])]

    # Belt and braces: also resume anything that still looks like a training
    # process, in case the control file was lost.
    for t in find_targets():
        if t.pid not in pids and process_alive(t.pid):
            pids.append(t.pid)

    if not pids:
        print("nothing to resume.")
    resumed: List[int] = []
    for pid in reversed(pids):
        if dry_run:
            print(f"  would resume pid {pid}")
            continue
        if not process_alive(pid):
            print(f"  pid {pid}: gone, skipping")
            continue
        try:
            resume(pid)
            resumed.append(pid)
            print(f"  resumed pid {pid}")
        except OSError as exc:
            print(f"  FAILED to resume pid {pid}: {exc}", file=sys.stderr)

    if dry_run:
        print("\n--dry-run: nothing changed.")
        return 0

    if resumed:
        active = still_running_hot(resumed, window=2.0, threshold=0.05)
        if active:
            print(f"  verified: activity resumed on {len(active)} process(es)")
        else:
            print("  note: no CPU activity yet -- normal if it just resumed "
                  "into a data-loading or checkpointing moment")

    if clear_gate():
        print(f"  gate cleared ({xs.GATE_PATH})")
    _write_control({"paused": False,
                    "resumed_at": xs._now().isoformat(timespec="seconds"),
                    "suspended_pids": []})
    print("resumed.")
    return 0


# ---------------------------------------------------------------------------
# Self-test: prove suspend/resume preserves work, on a throwaway process
# ---------------------------------------------------------------------------


def _read_last_counter(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").split()
        return int(lines[-1]) if lines else 0
    except (OSError, ValueError):
        return 0


def self_test() -> int:
    """Prove suspend/resume freezes the work and loses none of it.

    Deliberately uses the same launcher-stub/child topology as the real run, so
    the test also proves that suspending the *subtree* is what makes a pause
    effective -- suspending only ``proc.pid`` would leave the child counting and
    the test would fail.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="vr-pause-selftest-"))
    counter = tmpdir / "counter.txt"
    code = (
        "import time\n"
        f"p = r'{counter}'\n"
        "i = 0\n"
        "with open(p, 'w') as f:\n"
        "    while True:\n"
        "        i += 1\n"
        "        f.write(str(i) + chr(10))\n"
        "        f.flush()\n"
        "        time.sleep(0.1)\n"
    )
    print("spawning a throwaway counter process...")
    proc = subprocess.Popen([sys.executable, "-c", code])
    failures: List[str] = []
    try:
        deadline = time.time() + 15
        while _read_last_counter(counter) < 5 and time.time() < deadline:
            time.sleep(0.1)
        before = _read_last_counter(counter)
        if before < 5:
            print("FAIL: dummy process never started counting")
            return 1
        print(f"  counting: {before} iterations")

        table = process_table()
        pids = subtree_pids([proc.pid], table)
        if not pids:
            pids = [proc.pid]
        print(f"  target subtree: {pids}")

        print("suspending the whole subtree...")
        suspended = []
        for pid in sorted(pids, key=lambda p: (p != proc.pid, p)):
            try:
                suspend(pid)
                suspended.append(pid)
            except OSError as exc:
                failures.append(f"suspend pid {pid}: {exc}")

        # Settle first. Enumerating processes costs ~1 s of wall time, so the
        # counter legitimately advances between the reading above and the moment
        # the suspend actually lands. Only the interval *after* that is a test
        # of the pause.
        time.sleep(1.0)
        during = _read_last_counter(counter)
        time.sleep(1.5)
        still = _read_last_counter(counter)
        if still != during:
            failures.append(f"advanced while suspended: {during} -> {still}")
        print(f"  {'ok  ' if still == during else 'FAIL'} counter frozen "
              f"({during} -> {still})")

        hot = still_running_hot(suspended, window=1.0, threshold=0.05)
        if hot:
            failures.append(f"CPU still burning while suspended: {hot}")
        print(f"  {'ok  ' if not hot else 'FAIL'} no CPU consumed while suspended "
              f"({len(suspended)} pid(s))")

        print("resuming...")
        for pid in reversed(suspended):
            try:
                resume(pid)
            except OSError as exc:
                failures.append(f"resume pid {pid}: {exc}")
        time.sleep(1.5)

        after = _read_last_counter(counter)
        if after <= still:
            failures.append(f"did not advance after resume: {still} -> {after}")
        print(f"  {'ok  ' if after > still else 'FAIL'} advanced "
              f"({still} -> {after})")

        # The real claim: not one iteration was lost across the pause.
        seen = [int(x) for x in counter.read_text(encoding="utf-8").split()]
        contiguous = seen == list(range(1, len(seen) + 1))
        if not contiguous:
            failures.append("counter sequence has a gap -- work was lost")
        print(f"  {'ok  ' if contiguous else 'FAIL'} contiguous 1..{len(seen)} "
              f"(no lost iterations)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        for p in tmpdir.glob("*"):
            p.unlink(missing_ok=True)
        tmpdir.rmdir()

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSELF-TEST PASSED: the subtree freezes and loses no work across the pause.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="status",
                    choices=["status", "pause", "resume", "targets"])
    ap.add_argument("--note", default="", help="Why the pause was requested.")
    ap.add_argument("--dry-run", action="store_true", help="Show, do not act.")
    ap.add_argument("--force", action="store_true",
                    help="Allow suspending processes that do not look like a trainer.")
    ap.add_argument("--json", action="store_true", help="JSON output (status).")
    ap.add_argument("--self-test", action="store_true",
                    help="Verify suspend/resume on a throwaway process.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.command == "targets":
        for t in find_targets():
            print(f"{t.pid}\t{t.ppid}\t{t.short}")
        return 0
    if args.command == "status":
        return cmd_status(args.json)
    if args.command == "pause":
        return cmd_pause(args.note, args.dry_run, args.force)
    return cmd_resume(args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
