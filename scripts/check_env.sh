#!/usr/bin/env bash
# Verify the BudgetDraft evaluation environment.
# Exit code 0 = ready to reproduce, 1 = something missing or wrong version.
#
# Usage: bash scripts/check_env.sh   (or `make check`)

set -u
fail=0

ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }
err()  { echo "  ✗ $1"; fail=1; }

echo "=== System ==="
uname -a
python3 --version 2>&1 | sed 's/^/  /'

echo
echo "=== CUDA / NVIDIA ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
else
  err "nvidia-smi not found — CUDA runtime missing"
fi

echo
echo "=== Python packages ==="
python3 - <<'PY'
import sys

def check(modname, want=None, comparator="startswith"):
    try:
        m = __import__(modname)
        v = getattr(m, "__version__", "?")
        if want is None:
            print(f"  ✓ {modname} = {v}")
            return True
        if comparator == "startswith":
            ok = v.startswith(want)
        elif comparator == "exact":
            ok = v == want
        else:
            ok = True
        sym = "✓" if ok else "⚠"
        msg = "" if ok else f" (expected {want}.x — may or may not work)"
        print(f"  {sym} {modname} = {v}{msg}")
        return ok
    except ImportError:
        print(f"  ✗ {modname} NOT INSTALLED")
        return False

# Hard requirements
ok = True
ok &= check("torch", "2.4")
ok &= check("transformers", "4.44")
ok &= check("datasets", "2.18", "exact") or check("datasets", "2.")  # 2.x accepted, ≥3.0 broken
ok &= check("accelerate")
ok &= check("sentencepiece")
ok &= check("huggingface_hub")
ok &= check("termcolor")
ok &= check("tqdm")
ok &= check("numpy")
# Optional but recommended
try:
    import flash_attn
    print(f"  ✓ flash_attn = {flash_attn.__version__}")
except ImportError:
    print("  ⚠ flash_attn NOT installed (optional, slower training)")

# Quick torch.cuda sanity
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"  ✓ torch.cuda OK — device {torch.cuda.get_device_name(0)}, capability {cc[0]}.{cc[1]}")
else:
    print("  ✗ torch.cuda.is_available() = False — training/eval will fail")
    ok = False

sys.exit(0 if ok else 1)
PY
py_ok=$?
[ "$py_ok" -eq 0 ] || fail=1

echo
echo "=== Data ==="
if [ -f data/pg19_test.jsonl ]; then
  ok "data/pg19_test.jsonl present ($(wc -l < data/pg19_test.jsonl) lines)"
else
  warn "data/pg19_test.jsonl missing — will fall back to HF datasets script for PG-19"
fi

echo
echo "=== Eval script has per-sample emission? ==="
if grep -q "_samples.csv" src/eval.py; then
  ok "src/eval.py has _samples.csv emission (commit 080b464 or later)"
else
  err "src/eval.py is pre-080b464 — error bars won't work. git pull origin main"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "✅ Environment looks good. Try: make smoke"
  exit 0
else
  echo "❌ Environment has issues — see README (env setup + troubleshooting)"
  exit 1
fi
