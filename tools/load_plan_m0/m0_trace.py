#!/usr/bin/env python3
"""M0 for vLLM RFC #64418 (Load plan): a TraceTensor prototype plus init report.

For one model family it answers: of the checkpoint tensors a sender would
declare, which names are DIRECT (every destination byte is a copy of one
incoming byte, through view ops only) and which are FALLBACK (go through
``load_weights`` as a full tensor), how many bytes each side holds, and how long
the bake takes.

Method (prototype, not the P1 design):
  * the vLLM model is built on the meta device (no GPU memory, no weights);
    parameters are therefore in checkpoint layout, as after view-back;
  * each declared HF tensor is a fresh meta tensor with its own storage
    (the "base"); a TorchFunctionMode watches every torch call that sees one;
  * ``copy_(dst, src)`` with a tracked src is the only recorded write; the
    destination parameter is found by storage id, the source box by view
    geometry (bounding box of the view's corner elements; must be rectangular);
  * view ops keep tracking; an op that allocates new storage from a tracked
    tensor (cast, arithmetic, copying reshape, gather), an in-place op on a
    source, a broadcast, a dtype cast at ``copy_``, a copy into a temporary,
    or a value read marks the name FALLBACK and the bake continues
    (a value read raises on meta: the name is dropped and the bake reruns).
Not in the prototype: module-state snapshot, overlap check, F2 required-set,
``view(dtype)`` reinterpretation, tied-weight sharing across storages.

Usage (inside the vLLM container, see run_m0.sh):
  python m0_trace.py --model-dir DIR --headers safetensors:DIR --tag NAME --out OUT.json
  python m0_trace.py --model-dir DIR --headers synth:fused|per_expert[:fp8-block|fp8-tensor] ...
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


# --------------------------------------------------------------------------- headers
def read_safetensors_headers(d: str) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    out = []
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            out.append((k, ST_DTYPES[v["dtype"]], tuple(v["shape"])))
    return out


def _hf_meta_state(model_dir: str):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    return [(k, v.dtype, tuple(v.shape)) for k, v in m.state_dict().items()]


def synth_headers(model_dir: str, spec: str):
    """Headers from a transformers meta model, then Hub-style rewrites.

    spec = "<experts>[:<quant>]"; experts: per_expert (Hub checkpoints) or fused
    (transformers v5 [E, 2I, H] / [E, H, I]); quant: fp8-block (weight fp8 +
    weight_scale_inv) or fp8-tensor (ModelOpt static: weight_scale, input_scale).
    """
    parts = spec.split(":")
    experts = parts[0]
    quant = parts[1] if len(parts) > 1 else None
    out = []
    for name, dt, sh in _hf_meta_state(model_dir):
        m = re.match(r"(.*\.experts)\.(gate_up_proj|down_proj)$", name)
        if m and experts == "per_expert":
            E = sh[0]
            for e in range(E):
                if m.group(2) == "gate_up_proj":
                    i2 = sh[1] // 2
                    out.append((f"{m.group(1)}.{e}.gate_proj.weight", dt, (i2, sh[2])))
                    out.append((f"{m.group(1)}.{e}.up_proj.weight", dt, (i2, sh[2])))
                else:
                    out.append((f"{m.group(1)}.{e}.down_proj.weight", dt, sh[1:]))
        else:
            out.append((name, dt, sh))
    if quant is None:
        return out
    res = []
    for name, dt, sh in out:
        is_lin = (
            name.endswith(".weight")
            and len(sh) == 2
            and not any(s in name for s in ("embed_tokens", "lm_head", "norm"))
            and not name.endswith(".mlp.gate.weight")
            and "visual" not in name
        )
        if not is_lin:
            res.append((name, dt, sh))
        elif quant == "fp8-block":
            res.append((name, torch.float8_e4m3fn, sh))
            res.append(
                (
                    name.replace(".weight", ".weight_scale_inv"),
                    torch.float32,
                    (-(-sh[0] // 128), -(-sh[1] // 128)),
                )
            )
        elif quant == "fp8-tensor":
            res.append((name, torch.float8_e4m3fn, sh))
            res.append((name.replace(".weight", ".weight_scale"), torch.float32, ()))
            res.append((name.replace(".weight", ".input_scale"), torch.float32, ()))
        else:
            raise ValueError(quant)
    return res


# --------------------------------------------------------------------------- model
def build_meta_model(model_dir: str):
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import initialize_model

    args = EngineArgs(
        model=model_dir,
        dtype="bfloat16",
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=2048,
        enforce_eager=True,
    )
    vllm_config = args.create_engine_config()
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{29500 + os.getpid() % 1000}",
        backend="gloo",
    )
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(1, 1)
        t0 = time.perf_counter()
        from vllm.utils.torch_utils import set_default_torch_dtype

        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with torch.device("meta"):
                model = initialize_model(vllm_config)
        build_s = time.perf_counter() - t0
    return model, vllm_config, build_s


# --------------------------------------------------------------------------- tracer
def sid(t: torch.Tensor) -> int:
    return t.untyped_storage()._cdata


VIEW_OK = {
    "narrow", "view", "view_as", "reshape", "reshape_as", "__getitem__", "unsqueeze",
    "squeeze", "transpose", "t", "permute", "flatten", "unflatten", "contiguous",
    "chunk", "split", "unbind", "to", "movedim", "swapaxes", "select", "T", "mT",
}  # fmt: skip


def _fname(func) -> str:
    n = getattr(func, "__name__", None)
    if n is None:
        n = getattr(getattr(func, "__self__", None), "__name__", repr(func))
    # property getters look like "__get__" on a descriptor
    owner = getattr(func, "__self__", None)
    if n == "__get__" and owner is not None:
        n = getattr(owner, "__name__", n)
    return n


def _tensors_in(args, kwargs):
    for a in (*args, *kwargs.values()):
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, (list, tuple)):
            for b in a:
                if isinstance(b, torch.Tensor):
                    yield b


class Tracer(TorchFunctionMode):
    def __init__(self, bases: dict, params: dict):
        super().__init__()
        self.bases = bases  # storage id -> (name, base tensor)
        self.params = params  # storage id -> param name
        self.flag: dict[str, str] = {}
        self.writes: dict[str, list] = defaultdict(list)
        self.last: str | None = None
        self.calls = 0
        self.mismatch_tracked_dst = 0

    # -- geometry
    def box_of(self, name: str, s: torch.Tensor):
        base = self.bases[sid(s)][1]
        if s.numel() == 0:
            return None
        bs = base.stride()
        o = s.storage_offset()
        sh, st = s.shape, s.stride()
        if any(x == 0 for x in st) and any(n > 1 for n in sh):
            return None
        lo = [10**18] * base.dim()
        hi = [-1] * base.dim()
        import itertools

        for corner in itertools.product(*[(0, n - 1) if n > 1 else (0,) for n in sh]):
            pos = o + sum(c * s_ for c, s_ in zip(corner, st))
            coords = []
            for k in range(base.dim()):
                coords.append(pos // bs[k] if bs[k] else 0)
                pos = pos % bs[k] if bs[k] else pos
            for k, c in enumerate(coords):
                lo[k] = min(lo[k], c)
                hi[k] = max(hi[k], c)
        vol = 1
        for k in range(base.dim()):
            vol *= hi[k] - lo[k] + 1
        if base.dim() == 0:
            return ()
        if vol != s.numel():
            return None
        return tuple((lo[k], hi[k] + 1) for k in range(base.dim()))

    def mark(self, name: str, reason: str):
        self.flag.setdefault(name, reason)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls += 1
        tracked = []
        for t in _tensors_in(args, kwargs):
            if t.is_meta:
                i = sid(t)
                if i in self.bases:
                    tracked.append((self.bases[i][0], t))
        if not tracked:
            return func(*args, **kwargs)
        name0 = tracked[0][0]
        self.last = name0
        fname = _fname(func)

        if func is torch.Tensor.copy_:
            dst, src = args[0], args[1]
            if not (isinstance(src, torch.Tensor) and src.is_meta and sid(src) in self.bases):
                self.mark(name0, "tracked tensor used as copy_ destination")
                return func(*args, **kwargs)
            nm = self.bases[sid(src)][0]
            pname = self.params.get(sid(dst))
            if pname is None:
                self.mark(nm, "copy_ into a temporary (not a model tensor)")
            elif dst.dtype != src.dtype:
                self.mark(nm, f"cast at copy_ ({src.dtype} -> {dst.dtype})")
            elif src.dtype != self.bases[sid(src)][1].dtype:
                self.mark(nm, "view(dtype) reinterpretation (not handled in prototype)")
            else:
                box = self.box_of(nm, src)
                if box is None:
                    self.mark(nm, "source is not a rectangular box (step/broadcast)")
                elif dst.numel() != src.numel():
                    self.mark(nm, "broadcasting copy_")
                else:
                    self.writes[nm].append(
                        dict(
                            param=pname,
                            box=box,
                            offset=dst.storage_offset(),
                            shape=tuple(dst.shape),
                            stride=tuple(dst.stride()),
                            nbytes=src.numel() * src.element_size(),
                        )
                    )
            return func(*args, **kwargs)

        if fname == "__setitem__":
            # the value is written by index assignment, which a tracer cannot
            # see as a copy_; this is what #64418 L3 removes
            self.mark(name0, "write via __setitem__ (not copy_)")
            return func(*args, **kwargs)

        out = func(*args, **kwargs)
        if len({n for n, _ in tracked}) > 1:
            self.mark(name0, "combines several checkpoint tensors")
        # in-place op on a tracked source
        if fname.endswith("_") and not fname.startswith("__") and tracked[0][1] is args[0]:
            self.mark(name0, f"in-place op {fname} on source")
        outs = [o for o in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(o, torch.Tensor)]
        for o in outs:
            if not o.is_meta:
                continue
            if sid(o) in self.bases:
                if any(n > 1 and s == 0 for n, s in zip(o.shape, o.stride())):
                    self.mark(name0, f"broadcast/expand via {fname}")
            else:
                # result owns new storage: a copy/cast/arithmetic/gather
                self.mark(name0, f"op allocates new data: {fname}")
        return out


def run_trace(model, headers, tag: str):
    params: dict[int, str] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        params.setdefault(sid(p.data), n)
    for n, b in model.named_buffers(remove_duplicate=False):
        if b.is_meta:
            params.setdefault(sid(b), n)

    dropped: dict[str, str] = {}
    t_total = 0.0
    reruns = 0
    calls = 0
    while True:
        bases = {}
        fakes = []
        for name, dt, sh in headers:
            if name in dropped:
                continue
            t = torch.empty(sh, dtype=dt, device="meta")
            bases[sid(t)] = (name, t)
            fakes.append((name, t))
        tr = Tracer(bases, params)
        t0 = time.perf_counter()
        try:
            with tr:
                model.load_weights(iter(fakes))
            t_total += time.perf_counter() - t0
            calls = tr.calls
            break
        except Exception as e:  # value read or loader refusing a meta tensor
            t_total += time.perf_counter() - t0
            culprit = tr.last
            if culprit is None or culprit in dropped or reruns > 200:
                raise
            dropped[culprit] = f"loader raised {type(e).__name__}: {str(e)[:80]}"
            reruns += 1

    rows = []
    for name, dt, sh in headers:
        nbytes = torch.empty(sh, dtype=dt, device="meta").numel() * torch.empty((), dtype=dt).element_size()
        if name in dropped:
            status, reason = "fallback", dropped[name]
        elif name in tr.flag:
            status, reason = "fallback", tr.flag[name]
        elif name in tr.writes:
            status, reason = "direct", ""
        else:
            status, reason = "skipped", "loader wrote nothing (tied/other rank/ignored)"
        rows.append(
            dict(
                name=name, dtype=str(dt).replace("torch.", ""), shape=list(sh),
                nbytes=nbytes, status=status, reason=reason,
                writes=len(tr.writes.get(name, [])),
            )
        )  # fmt: skip
    return rows, dict(bake_s=t_total, reruns=reruns, torch_calls=calls)


def summarize(rows):
    def norm(n):
        return re.sub(r"\.\d+\.", ".*.", re.sub(r"\.\d+\.", ".*.", n))

    agg = collections.OrderedDict()
    for st in ("direct", "fallback", "skipped"):
        sel = [r for r in rows if r["status"] == st]
        agg[st] = dict(names=len(sel), bytes=sum(r["nbytes"] for r in sel))
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["status"] != "direct":
            k = (r["status"], norm(r["name"]), r["reason"])
            groups[k][0] += 1
            groups[k][1] += r["nbytes"]
    gl = [
        dict(status=k[0], pattern=k[1], reason=k[2], names=v[0], bytes=v[1])
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1][1])
    ]
    return agg, gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--headers", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kind, _, spec = a.headers.partition(":")
    t0 = time.perf_counter()
    headers = (
        read_safetensors_headers(spec) if kind == "safetensors" else synth_headers(a.model_dir, spec)
    )
    hdr_s = time.perf_counter() - t0

    model, vcfg, build_s = build_meta_model(a.model_dir)
    rows, meta = run_trace(model, headers, a.tag)
    agg, groups = summarize(rows)
    import vllm

    res = dict(
        tag=a.tag, vllm=vllm.__version__, headers=a.headers, n_names=len(rows),
        total_bytes=sum(r["nbytes"] for r in rows), summary=agg, groups=groups,
        header_s=hdr_s, model_build_meta_s=build_s, **meta,
    )  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(result=res, rows=rows), f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "groups"}, indent=1))
    for g in groups[:12]:
        print(g)


if __name__ == "__main__":
    main()
