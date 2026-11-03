#!/usr/bin/env python3
"""Cold-start a model, snapshot every parameter/buffer (sha256 + data_ptr), and
optionally run a direct reload of the same checkpoint and snapshot again.

  VLLM_RELOAD_MODE=direct python l3_verify.py --model-dir D --out J.json [--reload]

Compare two such JSONs (unpatched vs patched vLLM) with compare_snapshots.py.
Hash is over the raw bytes of each tensor (uint8 view), so it is exact.
"""
import argparse
import hashlib
import json
import time

import torch


def snapshot(model):
    snap = {}
    for kind, it in (
        ("p", model.named_parameters(remove_duplicate=False)),
        ("b", model.named_buffers(remove_duplicate=False)),
    ):
        for n, t in it:
            t = t.data
            flat = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            snap[n] = dict(
                kind=kind, dtype=str(t.dtype), shape=list(t.shape),
                sha=hashlib.sha256(flat.tobytes()).hexdigest(), ptr=t.data_ptr(),
            )  # fmt: skip
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.reload import finish_reload, start_reload

    args = EngineArgs(
        model=a.model_dir, dtype="bfloat16", skip_tokenizer_init=True,
        max_model_len=2048, enforce_eager=True,
    )  # fmt: skip
    vcfg = args.create_engine_config()
    mcfg = vcfg.model_config
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29611", backend="gloo",
    )  # fmt: skip
    loader = get_model_loader(vcfg.load_config)
    res = {}
    with set_current_vllm_config(vcfg):
        ensure_model_parallel_initialized(1, 1)
        t0 = time.perf_counter()
        model = loader.load_model(vcfg, mcfg)
        res["cold_s"] = time.perf_counter() - t0
        res["cold"] = snapshot(model)
        if a.reload:
            dev = torch.device("cuda")
            start_reload(model)
            t0 = time.perf_counter()
            model.load_weights(loader.get_all_weights(mcfg, model))
            finish_reload(model, mcfg, dev)
            torch.accelerator.synchronize()
            res["reload_s"] = time.perf_counter() - t0
            res["reloaded"] = snapshot(model)
            c, r = res["cold"], res["reloaded"]
            diff_sha = [n for n in c if c[n]["sha"] != r.get(n, {}).get("sha")]
            diff_ptr = [n for n in c if c[n]["ptr"] != r.get(n, {}).get("ptr")]
            res["reload_vs_cold"] = dict(
                n_tensors=len(c), bytes_differ=diff_sha[:20],
                n_bytes_differ=len(diff_sha), addr_moved=diff_ptr[:20],
                n_addr_moved=len(diff_ptr),
            )  # fmt: skip
            print("reload_vs_cold", json.dumps(res["reload_vs_cold"]))
    import vllm

    res["vllm"] = vllm.__version__
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("wrote", a.out, "tensors", len(res["cold"]))


if __name__ == "__main__":
    main()
