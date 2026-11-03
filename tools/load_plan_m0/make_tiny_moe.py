#!/usr/bin/env python3
"""Write a random-init tiny Qwen3-MoE checkpoint (safetensors) in the Hub
per-expert layout. quant=none: BF16; quant=fp8-tensor: ModelOpt static per-tensor
FP8 (weight_scale + input_scale per Linear and per expert). No download.

  python make_tiny_moe.py OUTDIR [none|fp8-tensor|fp8-block]
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from m0_trace import synth_headers  # noqa: E402

out = Path(sys.argv[1])
quant = sys.argv[2] if len(sys.argv) > 2 else "none"
out.mkdir(parents=True, exist_ok=True)
cfg = {
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
    "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 128,
    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 64, "num_experts": 8, "num_experts_per_tok": 2, "vocab_size": 1024,
    "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [],
    "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "max_position_embeddings": 2048, "tie_word_embeddings": False,
    "attention_bias": False, "torch_dtype": "bfloat16",
}  # fmt: skip
if quant == "fp8-tensor":
    cfg["quantization_config"] = {
        "quant_method": "modelopt", "quant_algo": "FP8",
        "exclude_modules": ["lm_head", "*mlp.gate"],
    }  # fmt: skip
elif quant == "fp8-block":
    cfg["quantization_config"] = {
        "quant_method": "fp8", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }  # fmt: skip
(out / "config.json").write_text(json.dumps(cfg, indent=1))
g = torch.Generator().manual_seed(0)
spec = "per_expert" + ("" if quant == "none" else ":" + quant)
tensors = {}
for name, dt, sh in synth_headers(str(out), spec):
    if dt == torch.float8_e4m3fn:
        raw = torch.randint(0, 0x78, sh, generator=g, dtype=torch.uint8)
        raw |= torch.randint(0, 2, sh, generator=g, dtype=torch.uint8) << 7
        tensors[name] = raw.view(torch.float8_e4m3fn)
    elif dt == torch.float32:
        tensors[name] = torch.rand(sh, generator=g) * 0.02 + 0.005
    else:
        tensors[name] = (torch.randn(sh, generator=g) * 0.05).to(dt)
save_file(tensors, str(out / "model.safetensors"))
print(out, len(tensors), "tensors")
