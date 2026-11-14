"""Residual-stream activations.

Layer convention (same as jlens): layer L = output of transformer block L
(0-based) = the residual stream after L+1 blocks. In Hugging Face terms that
is `output_hidden_states[L + 1]`, except for the last layer, where HF applies
the final norm and we do not. Layer n_layers-1 is the last block's output
before the final norm.

Positions: "last" (last real token), "mean" (mean over real tokens), "all"
(list of [seq_len, d_model] tensors, one per prompt), or an int (Python index
into each prompt's real tokens, e.g. -2).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ACTS_VERSION = 2


def cache_dir() -> Path:
    d = Path(os.environ.get("LAB_CACHE", Path.home() / ".cache" / "lab")) / "acts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _versions() -> dict:
    out = {}
    for mod in ("torch", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _key(model, prompts, layers, position, max_length: int = 512) -> str:
    """Everything that changes the numbers goes into the key: the checkpoint's commit on the Hub
    (None for a local folder), dtype, truncation length and the library versions."""
    revision = getattr(getattr(getattr(model, "hf", None), "config", None), "_commit_hash", None)
    blob = json.dumps({"v": ACTS_VERSION, "model": model.name, "revision": revision,
                       "dtype": str(model.dtype), "libs": _versions(), "max_length": max_length,
                       "prompts": list(prompts), "layers": list(layers),
                       "position": position}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def run_body(model, enc):
    """Run the decoder stack (no LM head) on an encoded batch."""
    return model.lm._text_module(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"], use_cache=False)


def get_acts(model, prompts, layers=None, position="last", batch_size: int = 8,
             cache: bool = True, max_length: int = 512, verbose: bool = False):
    """Return {layer: tensor}. For "last"/"mean"/int: float32 CPU tensor
    [n_prompts, d_model]. For "all": list of [seq_len_i, d_model] tensors.
    Results are cached on disk under ~/.cache/lab/acts keyed by a hash of
    (model, its Hub commit, dtype, prompts, layers, position, max_length and
    the torch and transformers versions)."""
    import torch

    if isinstance(prompts, str):
        prompts = [prompts]
    layers = list(range(model.n_layers)) if layers is None else [
        l % model.n_layers for l in layers]
    path = cache_dir() / f"{_key(model, prompts, layers, position, max_length)}.pt"
    if cache and path.exists():
        if verbose:
            print(f"[lab] acts from cache {path.name}")
        return torch.load(path, weights_only=True)

    out = {l: [] for l in layers}
    store = {}
    handles = []
    for l in layers:
        def hook(module, inputs, output, l=l):
            store[l] = (output if torch.is_tensor(output) else output[0]).detach()
        handles.append(model.layers[l].register_forward_hook(hook))
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                enc = model.encode(prompts[i:i + batch_size], max_length=max_length)
                store.clear()
                run_body(model, enc)
                for l in layers:
                    h = store[l].float()
                    for b, n in enumerate(enc["lengths"]):
                        hb = h[b, :n]
                        if position == "last":
                            out[l].append(hb[-1].cpu())
                        elif position == "mean":
                            out[l].append(hb.mean(0).cpu())
                        elif position == "all":
                            out[l].append(hb.cpu())
                        elif isinstance(position, int):
                            out[l].append(hb[position].cpu())
                        else:
                            raise ValueError(f"bad position {position!r}")
    finally:
        for h in handles:
            h.remove()
    res = {l: (v if position == "all" else torch.stack(v)) for l, v in out.items()}
    if cache:
        torch.save(res, path)
    return res
