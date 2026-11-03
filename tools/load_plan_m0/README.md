# Load plan M0 prototype (vLLM RFC #64418)

TraceTensor prototype for the M0 item of #64418: for one model family, report which
checkpoint names a load plan could write **directly** (every destination byte is a
`copy_` of one incoming byte through view ops only) and which **fall back** to
`load_weights`. Method and limitations are in the docstring of `m0_trace.py`.
The model is built on the meta device, so no GPU memory or real weights are needed.

| file | purpose |
|---|---|
| `m0_trace.py` | the tracer + report (JSON) |
| `m0_table.py` | markdown table from result JSONs |
| `m0_cases.sh` | the cases below (runs inside the `vllm/vllm-openai:nightly` image, models under `/models`) |
| `make_tiny_moe.py`, `l3_verify.py`, `l3_cases.sh`, `compare_snapshots.py` | L3 check: cold-start sha256 of every tensor, and `VLLM_RELOAD_MODE=direct` reload vs cold start |

Results on nightly `e0d00e4b8`, TP=1 (`synth:` = synthetic safetensors header with real shapes, no weights).
`-base` = unmodified, `-patched` = with the L3 change (#65259).

| case | names | GiB | direct names | direct GiB | fallback names | fallback GiB | skipped | bake s | torch calls | reruns |
|---|---|---|---|---|---|---|---|---|---|---|
| q06_bf16-base | 311 | 1.40 | 310 | 1.11 | 0 | 0.00 | 1 | 0.03 | 2625 | 0 |
| q06_fp8-base | 507 | 0.99 | 310 | 0.70 | 196 | 0.00 | 1 | 0.06 | 4557 | 0 |
| q30_bf16_fused-base | 531 | 56.87 | 531 | 56.87 | 0 | 0.00 | 0 | 2.79 | 366418 | 0 |
| q30_bf16_perexpert-base | 18867 | 56.87 | 18867 | 56.87 | 0 | 0.00 | 0 | 3.22 | 421378 | 0 |
| q30_fp8block-base | 37491 | 29.03 | 37491 | 29.03 | 0 | 0.00 | 0 | 6.90 | 840802 | 0 |
| q30_fp8tensor-base | 56115 | 29.03 | 19251 | 29.03 | 36864 | 0.00 | 0 | 7.59 | 842242 | 0 |
| q35_4b-base | 738 | 8.68 | 675 | 8.45 | 48 | 0.00 | 15 | 0.37 | 7640 | 24 |
| q8_bf16-base | 399 | 15.26 | 399 | 15.26 | 0 | 0.00 | 0 | 0.04 | 3382 | 0 |
| q8_fp8-base | 651 | 8.79 | 399 | 8.79 | 252 | 0.00 | 0 | 0.07 | 5866 | 0 |
| q06_bf16-patched | 311 | 1.40 | 310 | 1.11 | 0 | 0.00 | 1 | 0.03 | 2625 | 0 |
| q06_fp8-patched | 507 | 0.99 | 310 | 0.70 | 196 | 0.00 | 1 | 0.06 | 4557 | 0 |
| q30_bf16_fused-patched | 531 | 56.87 | 531 | 56.87 | 0 | 0.00 | 0 | 2.67 | 366418 | 0 |
| q30_bf16_perexpert-patched | 18867 | 56.87 | 18867 | 56.87 | 0 | 0.00 | 0 | 3.61 | 421378 | 0 |
| q30_fp8block-patched | 37491 | 29.03 | 37491 | 29.03 | 0 | 0.00 | 0 | 6.84 | 840802 | 0 |
| q30_fp8tensor-patched | 56115 | 29.03 | 56115 | 29.03 | 0 | 0.00 | 0 | 8.44 | 879106 | 0 |
| q35_4b-patched | 738 | 8.68 | 699 | 8.46 | 24 | 0.00 | 15 | 0.45 | 7568 | 24 |
| q8_bf16-patched | 399 | 15.26 | 399 | 15.26 | 0 | 0.00 | 0 | 0.04 | 3382 | 0 |
| q8_fp8-patched | 651 | 8.79 | 399 | 8.79 | 252 | 0.00 | 0 | 0.08 | 5866 | 0 |

Notes: Qwen3-8B-FP8 / 0.6B-FP8 fallbacks are bf16 `weight_scale_inv` cast to fp32 at load (L-item, not L3).
The 24 remaining Qwen3.5-4B fallbacks are `linear_attn.norm.weight` (separate cause).
Fallback bytes are < 0.1% in every case: these are scalar scales, but each one still forces the slow path for its name.

AI assistance (Claude Code) was used to write this prototype.
