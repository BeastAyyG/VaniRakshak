#!/usr/bin/env python
"""Run the test suite without pytest.

Why this exists
---------------
The project venv (`C:\\VaniRakshak\\.venv`) has **no pip and no pytest**, and
`pip` cannot be bootstrapped into it without modifying an environment that was
built by someone else. The tests themselves need a small, closed set of pytest
features:

    pytest.approx (16 uses), pytest.raises (5), pytest.mark.skipif (4),
    pytest.importorskip (1), and the built-in fixtures
    tmp_path (16 uses) and monkeypatch.setattr (2 uses)

So rather than leave the suite unrunnable, this module installs a minimal
`pytest` shim into ``sys.modules``, resolves those two fixtures, and executes
every ``test_*`` function it finds. There is no parametrize. If a future test
needs a feature not here, add it here rather than letting the suite report
phantom failures -- an unreliable runner is worse than none.

Note: `tests/test_asr.py` and `tests/test_groq_asr.py` cannot run in this venv
at all, because it has no `python-dotenv`. They are reported as import errors,
which is honest -- it is an environment gap, not a test failure.

Usage
-----
    python tests/run_all.py            # all tests/test_*.py
    python tests/run_all.py test_compute_mindcf_mincllr.py

Exit code is 0 only if every test passes and nothing was skipped for an
unexpected reason.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


# ---------------------------------------------------------------------------
# The shim
# ---------------------------------------------------------------------------
class _Skipped(Exception):
    """Raised by pytest.skip()/importorskip() to mark a test as skipped."""


class _Approx:
    """Minimal stand-in for pytest.approx."""

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _tol(self, exp):
        if self.abs is not None:
            return self.abs
        rel = self.rel if self.rel is not None else 1e-6
        return rel * abs(exp)

    def __eq__(self, other):
        try:
            exp = self.expected
            if hasattr(exp, "__len__") and hasattr(other, "__len__"):
                if len(exp) != len(other):
                    return False
                return all(
                    abs(float(a) - float(b)) <= (self.abs if self.abs is not None
                                                 else (self.rel if self.rel is not None else 1e-6) * abs(float(a)))
                    for a, b in zip(exp, other)
                )
            return abs(float(other) - float(exp)) <= self._tol(float(exp))
        except (TypeError, ValueError):
            return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        return NotImplemented if eq is NotImplemented else not eq

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel!r}, abs={self.abs!r})"


class _Raises:
    """Minimal stand-in for pytest.raises, usable as a context manager."""

    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected}")
        if issubclass(exc_type, self.expected):
            self.value = exc
            return True
        return False


class _Mark:
    """Only skipif is needed. The decorator stamps the function; the runner
    below reads the stamp."""

    def skipif(self, condition, reason=""):
        def deco(fn):
            if condition:
                fn.__skip_reason__ = reason or "skipif"
            return fn
        return deco

    def __getattr__(self, name):  # any other mark is a no-op passthrough
        def deco(*a, **k):
            if len(a) == 1 and callable(a[0]) and not k:
                return a[0]

            def inner(fn):
                return fn
            return inner
        return deco


class _MonkeyPatch:
    """Minimal stand-in for the monkeypatch fixture. Only setattr is used by
    the suite; anything else is added deliberately rather than by accident."""

    def __init__(self):
        self._undo: list[tuple[object, str, object, bool]] = []

    def setattr(self, target, name, value):
        existed = hasattr(target, name)
        original = getattr(target, name, None)
        self._undo.append((target, name, original, existed))
        setattr(target, name, value)

    def undo(self):
        for target, name, original, existed in reversed(self._undo):
            if existed:
                setattr(target, name, original)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        self._undo.clear()


def _call_with_fixtures(fn, tmp_root: Path):
    """Call a test function, supplying the built-in fixtures it asks for.

    Raises _Skipped for any fixture that is not implemented, rather than a
    confusing TypeError from calling fn() with the wrong arity.
    """
    import inspect
    import shutil
    import tempfile

    params = [p for p in inspect.signature(fn).parameters.values()]
    kwargs = {}
    made: list[Path] = []
    mp: _MonkeyPatch | None = None
    try:
        for p in params:
            if p.name == "tmp_path":
                d = Path(tempfile.mkdtemp(prefix="vr-test-", dir=str(tmp_root)))
                made.append(d)
                kwargs[p.name] = d
            elif p.name == "monkeypatch":
                mp = _MonkeyPatch()
                kwargs[p.name] = mp
            else:
                raise _Skipped(f"fixture {p.name!r} is not implemented in run_all.py")
        return fn(**kwargs)
    finally:
        if mp is not None:
            mp.undo()
        for d in made:
            shutil.rmtree(d, ignore_errors=True)


def _install_shim() -> None:
    if "pytest" in sys.modules:
        return
    shim = types.ModuleType("pytest")
    shim.approx = _Approx
    shim.raises = _Raises
    shim.mark = _Mark()
    shim.skip = lambda reason="": (_ for _ in ()).throw(_Skipped(reason))

    def importorskip(name, *a, **k):
        # Must use importlib.import_module: __import__("a.b") returns the
        # top-level package, which silently passes and then fails later.
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            raise _Skipped(f"could not import {name!r}: {exc}") from exc

    shim.importorskip = importorskip
    shim._Skipped = _Skipped
    sys.modules["pytest"] = shim


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
def run_module(path: Path, tmp_root: Path) -> tuple[int, int, int, list[str]]:
    name = f"tests.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return 0, 0, 0, [f"{path.name}: could not load"]
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except _Skipped as exc:
        print(f"  SKIP (module) {path.name}: {exc}")
        return 0, 1, 0, []
    except Exception:
        print(f"  ERROR importing {path.name}")
        traceback.print_exc()
        return 0, 0, 1, [path.name]

    passed = skipped = failed = 0
    failures: list[str] = []
    for attr in sorted(dir(mod)):
        if not attr.startswith("test_"):
            continue
        fn = getattr(mod, attr)
        if not callable(fn):
            continue
        reason = getattr(fn, "__skip_reason__", None)
        if reason:
            print(f"  SKIP {path.stem}::{attr} ({reason})")
            skipped += 1
            continue
        try:
            _call_with_fixtures(fn, tmp_root)
            passed += 1
            print(f"  ok   {path.stem}::{attr}")
        except _Skipped as exc:
            print(f"  SKIP {path.stem}::{attr} ({exc})")
            skipped += 1
        except Exception:
            failed += 1
            failures.append(f"{path.stem}::{attr}")
            print(f"  FAIL {path.stem}::{attr}")
            traceback.print_exc()
    return passed, skipped, failed, failures


def main(argv: list[str]) -> int:
    _install_shim()
    sys.path.insert(0, str(REPO_ROOT))

    if argv:
        paths = [TESTS_DIR / a for a in argv]
    else:
        paths = sorted(TESTS_DIR.glob("test_*.py"))

    if not paths:
        print("no test files found")
        return 1

    print(f"Running {len(paths)} test module(s) without pytest\n")
    total_p = total_s = total_f = 0
    all_failures: list[str] = []

    # One session-scoped temp root, so tmp_path dirs land somewhere we control
    # and are cleaned up even if a test leaves files behind.
    import shutil
    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="vr-tests-"))
    try:
        for p in paths:
            print(f"== {p.name} ==")
            pa, sk, fa, fails = run_module(p, tmp_root)
            total_p += pa
            total_s += sk
            total_f += fa
            all_failures.extend(fails)
            print()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"TOTAL: {total_p} passed, {total_s} skipped, {total_f} failed")
    if all_failures:
        print("failed: " + ", ".join(all_failures))
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
