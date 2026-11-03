#!/usr/bin/env python3
"""python compare_snapshots.py A.json B.json [key]  -- exact sha comparison."""
import json
import sys

with open(sys.argv[1]) as f:
    a = json.load(f)
with open(sys.argv[2]) as f:
    b = json.load(f)
key = sys.argv[3] if len(sys.argv) > 3 else "cold"
x, y = a[key], b[key]
only = sorted(set(x) ^ set(y))
diff = [n for n in x if n in y and x[n]["sha"] != y[n]["sha"]]
print(
    f"{sys.argv[1]} vs {sys.argv[2]} [{key}]: {len(x)} vs {len(y)} tensors, "
    f"name-set diff {len(only)}, byte diff {len(diff)}"
)
print("IDENTICAL" if not only and not diff else f"DIFF e.g. {only[:5]} {diff[:5]}")
sys.exit(0 if not only and not diff else 1)
