#!/usr/bin/env bash
# Runs inside the container (/lp mounted, /out writable). Usage: m0_cases.sh <label>
# Every case is independent; a failing case is logged and the rest continue.
LABEL=${1:-base}
M=/models
C=/cfg
cd /lp || exit 1
run() { # tag model-dir headers
  echo "=== $1"
  python3 m0_trace.py --model-dir "$2" --headers "$3" --tag "$1-$LABEL" --out "/out/$1-$LABEL.json" 2>&1 | grep -v "INFO\|WARNING\|^W1102" | tail -${TAILN:-6} || true
}
# modelopt per-tensor FP8 config derived from the Qwen3-30B-A3B config
mkdir -p /out/cfg_q30_modelopt
python3 - <<'EOF'
import json
c = json.load(open("/cfg/Qwen_Qwen3-30B-A3B/config.json"))
c["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "FP8",
                            "exclude_modules": ["lm_head", "*mlp.gate"]}
json.dump(c, open("/out/cfg_q30_modelopt/config.json", "w"))
EOF
run q06_bf16     $M/Qwen__Qwen3-0.6B       safetensors:$M/Qwen__Qwen3-0.6B
run q06_fp8      $M/Qwen__Qwen3-0.6B-FP8   safetensors:$M/Qwen__Qwen3-0.6B-FP8
run q8_bf16      $M/Qwen__Qwen3-8B         synth:per_expert
run q8_fp8       $M/Qwen__Qwen3-8B-FP8     safetensors:$M/Qwen__Qwen3-8B-FP8
run q35_4b       $M/Qwen__Qwen3.5-4B       safetensors:$M/Qwen__Qwen3.5-4B
run q30_bf16_perexpert $C/Qwen_Qwen3-30B-A3B synth:per_expert
run q30_bf16_fused     $C/Qwen_Qwen3-30B-A3B synth:fused
run q30_fp8block       $C/Qwen_Qwen3-30B-A3B-FP8 synth:per_expert:fp8-block
run q30_fp8tensor      /out/cfg_q30_modelopt synth:per_expert:fp8-tensor
