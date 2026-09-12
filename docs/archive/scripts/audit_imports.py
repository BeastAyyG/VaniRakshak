"""Audit helper: build a real import graph for the VaniRakshak repo.

Handles the two sys.path roots that the project injects:
  * REPO_ROOT            -> `backend.*`, `benchmarks.*`, `ml.*`
  * REPO_ROOT/vanirakshak-backend -> `vanirakshak.*`
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

ROOT = os.getcwd()
SKIP = {".git", "node_modules", "__pycache__", ".venv", ".workbuddy-ai", ".next"}

pyfiles = []
for root, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        if f.endswith(".py"):
            pyfiles.append(os.path.join(root, f))


def importable_names(path: str) -> list[str]:
    """Every dotted module name this file could legitimately be imported as."""
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    names = []
    # root-level layout
    m = rel[:-3].replace("/", ".")
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    names.append(m)
    # vanirakshak-backend/ is itself added to sys.path
    if rel.startswith("vanirakshak-backend/"):
        m2 = rel[len("vanirakshak-backend/") : -3].replace("/", ".")
        if m2.endswith(".__init__"):
            m2 = m2[: -len(".__init__")]
        names.append(m2)
    return names


src = {}
for p in pyfiles:
    try:
        src[p] = open(p, errors="replace").read()
    except Exception:
        src[p] = ""

name_to_file: dict[str, str] = {}
for p in pyfiles:
    for n in importable_names(p):
        name_to_file.setdefault(n, p)

# count importers, only counting the FIRST (canonical) name of each file
importers = defaultdict(set)
for p, s in src.items():
    for mm in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", s, re.M):
        target = mm.group(1)
        # resolve to a file, walking up package prefixes
        cur = target
        while cur:
            f = name_to_file.get(cur)
            if f and f != p:
                importers[f].add(p)
                break
            cur = cur.rsplit(".", 1)[0] if "." in cur else ""

print("=" * 78)
print("IMPORT GRAPH  (0 importers = nothing in the repo imports it)")
print("=" * 78)
rows = []
for p in pyfiles:
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    rows.append((len(importers[p]), rel))
rows.sort()
for n, rel in rows:
    flag = "  <== DEAD" if n == 0 else ""
    print(f"{n:3d}  {rel}{flag}")

print()
print("=" * 78)
print("SCORED PATH: what does the benchmark actually import?")
print("=" * 78)
for p in pyfiles:
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    if rel in ("benchmarks/run_benchmarks.py", "benchmarks/compute_mindcf.py"):
        print(f"--- {rel} ---")
        for mm in re.finditer(r"^\s*(?:from|import)\s+\S+.*$", src[p], re.M):
            print("   ", mm.group(0).strip())

print()
print("=" * 78)
print("Does anything outside backend/ import the heavy `backend` package?")
print("=" * 78)
for p, s in src.items():
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    if rel.startswith("backend/"):
        continue
    for mm in re.finditer(r"^\s*(?:from|import)\s+backend\b.*$", s, re.M):
        print(f"  {rel}: {mm.group(0).strip()}")
