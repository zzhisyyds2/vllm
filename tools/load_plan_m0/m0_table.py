#!/usr/bin/env python3
"""Print the M0 table (markdown) from result JSONs. python m0_table.py DIR..."""
import glob
import json
import sys

rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/*.json")):
        r = json.load(open(f))["result"]
        s = r["summary"]
        gb = lambda x: f"{x/2**30:.2f}"  # noqa: E731
        rows.append((r["tag"], r["n_names"], gb(r["total_bytes"]), s["direct"]["names"], gb(s["direct"]["bytes"]),
                     s["fallback"]["names"], gb(s["fallback"]["bytes"]), s["skipped"]["names"], f'{r["bake_s"]:.2f}', r["torch_calls"], r["reruns"]))
print("| case | names | GiB | direct names | direct GiB | fallback names | fallback GiB | skipped | bake s | torch calls | reruns |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print("| " + " | ".join(map(str, x)) + " |")
