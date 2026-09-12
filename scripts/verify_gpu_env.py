"""Phase 0 Gate: Verify PyTorch CUDA installation, Blackwell sm_120 support, and bfloat16 matmul.

Follows docs/08-cm-training-plan.md §2.
"""

import sys

try:
    import torch
except ImportError as e:
    print(f"FAILED: torch not installed: {e}")
    sys.exit(1)

print("=" * 60)
print("Phase 0 — Environment & GPU Verification Gate")
print("=" * 60)
print(f"torch version : {torch.__version__}")
print(f"torch cuda ver: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("FAILED: torch.cuda.is_available() is False!")
    sys.exit(1)

device_count = torch.cuda.device_count()
print(f"device count  : {device_count}")

maj, minr = torch.cuda.get_device_capability(0)
device_name = torch.cuda.get_device_name(0)
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
arch_list = torch.cuda.get_arch_list()

print(f"device        : {device_name}")
print(f"capability    : sm_{maj}{minr}")
print(f"vram GB       : {vram_gb} GB")
print(f"arch list     : {arch_list}")

# Gate: Blackwell sm_120 check
has_sm120 = any("120" in arch for arch in arch_list)
print(f"sm_120 in arch: {has_sm120}")

# Gate: bfloat16 matmul on device
print("Running 4096 x 4096 bfloat16 matmul on CUDA...")
try:
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    matmul_ok = bool(torch.isfinite(y).all())
    print(f"matmul ok     : {matmul_ok}")
    if not matmul_ok:
        print("FAILED: bfloat16 matmul produced NaNs or Infs!")
        sys.exit(1)
except Exception as e:
    print(f"FAILED: matmul raised exception: {e}")
    sys.exit(1)

print("=" * 60)
print("GATE PASSED: Blackwell RTX 5060 Ti CUDA environment is fully operational!")
print("=" * 60)
