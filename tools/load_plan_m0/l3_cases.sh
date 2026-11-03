#!/usr/bin/env bash
# inside container: l3_cases.sh <label>; cold-start hash + direct-reload check
L=${1:-base}
cd /lp || exit 1
[ -f /out/tiny_fp8t/model.safetensors ] || python3 make_tiny_moe.py /out/tiny_fp8t fp8-tensor 2>&1 | tail -1
export VLLM_RELOAD_MODE=direct
for c in tiny_fp8t:/out/tiny_fp8t q35_4b:/models/Qwen__Qwen3.5-4B; do
  n=${c%%:*}; d=${c#*:}
  echo "=== $n $L"
  python3 l3_verify.py --model-dir "$d" --out "/out/l3_${n}_${L}.json" --reload 2>&1 | grep -v "INFO\|WARNING\|^W1102" | tail -8
done
