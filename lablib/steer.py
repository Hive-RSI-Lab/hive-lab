"""Activation steering: add alpha * direction to the residual stream.

The vector is added to the OUTPUT of block `layer` (lablib/jlens layer
convention). `alpha * vector` is added as given: pass a unit direction and an
alpha in residual-norm units, or a raw difference-of-means vector with
alpha=1. Typical residual norms are printed by `residual_norm()`.

Positions: None = every position, including tokens produced during
generation. A list of ints = those prompt positions only (negative ints count
from each prompt's last real token); then generated tokens are not steered.

The built-in control: `compare_to_random` runs the same intervention with K
random directions of the same norm and reports where the real effect falls
in that null distribution. An effect that random directions also produce is
not evidence that the direction means anything.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from . import stats


class _Steer:
    def __init__(self, vector, alpha, positions):
        self.vector, self.alpha, self.positions = vector, alpha, positions
        self.lengths = None  # set per batch by eval helpers (right padding)

    def __call__(self, h):
        v = (self.alpha * self.vector).to(h.device, h.dtype)
        if self.positions is None:
            return h + v
        if h.shape[1] == 1:  # cached generation step: not a prompt position
            return None
        h = h.clone()
        lengths = self.lengths or [h.shape[1]] * h.shape[0]
        for b, n in enumerate(lengths):
            for p in self.positions:
                h[b, p if p >= 0 else n + p] += v
        return h


@contextmanager
def steering(model, layer: int, vector, alpha: float = 1.0, positions=None):
    """Context manager: while active, block `layer` outputs h + alpha*vector."""
    import torch

    from .models import block_hook

    vec = torch.as_tensor(vector, dtype=torch.float32).flatten()
    st = _Steer(vec, alpha, positions)
    with block_hook(model, layer % model.n_layers, st):
        yield st


def random_like(vector, k: int, seed: int = 0):
    """K random Gaussian directions with the same L2 norm as `vector`: [k, d]."""
    import torch

    v = torch.as_tensor(vector, dtype=torch.float32).flatten()
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(k, v.numel(), generator=g)
    return r / r.norm(dim=1, keepdim=True) * v.norm()


def last_logits(model, prompts, ctx=None, batch_size: int = 8):
    """Last-real-token logits [n, vocab]; tells a steering/ablation context the
    per-prompt lengths so negative positions are right under padding."""
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = model.encode(prompts[i:i + batch_size])
            if ctx is not None:
                ctx.lengths = enc["lengths"]
            lg = model.hf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                          use_cache=False).logits
            idx = torch.tensor(enc["lengths"], device=lg.device) - 1
            outs.append(lg[torch.arange(len(idx)), idx].float().cpu())
    return torch.cat(outs)


def logit_diff_metric(model, pos: str, neg: str | None = None):
    """Metric: log p(pos) - log p(neg) at the last token (or just log p(pos)).
    Returns f(logits [n, vocab]) -> numpy [n]."""
    import torch

    ip = model.token_id(pos)
    ineg = model.token_id(neg) if neg is not None else None

    def f(logits):
        lp = torch.log_softmax(logits.float(), -1)
        v = lp[:, ip] - (lp[:, ineg] if ineg is not None else 0)
        return v.numpy()

    return f


def residual_norm(model, prompts, layer: int) -> float:
    """Mean L2 norm of the last-token residual at `layer` (for choosing alpha)."""
    from .acts import get_acts

    a = get_acts(model, prompts, [layer], "last")[layer % model.n_layers]
    return float(a.norm(dim=1).mean())


def effect(model, prompts, layer, vector, alpha, metric, positions=None, base=None):
    """Per-prompt change in metric under steering (steered - unsteered)."""
    if base is None:
        base = metric(last_logits(model, prompts))
    with steering(model, layer, vector, alpha, positions) as st:
        steered = metric(last_logits(model, prompts, st))
    return steered - base


def compare_to_random(model, prompts, layer, vector, alpha, metric, k: int = 20,
                      positions=None, seed: int = 0):
    """Effect of steering along `vector` vs K random same-norm vectors.

    Returns dict: effect (mean change), effect_ci, null (K mean changes),
    null_mean, null_sd, z, p (two-sided empirical p against the null), and
    per_prompt (array). Each random vector costs one batched forward pass."""
    base = metric(last_logits(model, prompts))
    per = effect(model, prompts, layer, vector, alpha, metric, positions, base)
    null = [float(np.mean(effect(model, prompts, layer, r, alpha, metric, positions, base)))
            for r in random_like(vector, k, seed)]
    obs, lo, hi = stats.bootstrap_ci(per, seed=seed)
    sd = float(np.std(null, ddof=1)) if k > 1 else float("nan")
    return {"layer": layer, "alpha": alpha, "effect": obs, "effect_ci": (lo, hi),
            "null_mean": float(np.mean(null)), "null_sd": sd,
            "z": (obs - float(np.mean(null))) / sd if sd > 0 else float("nan"),
            "p": stats.empirical_p(obs - float(np.mean(null)),
                                   np.array(null) - float(np.mean(null))),
            "null": null, "per_prompt": per.tolist(), "k": k}


COMPARE_COLS = ["layer", "alpha", "effect", "effect_ci", "null_mean", "null_sd", "z", "p"]


def generate(model, prompt: str, layer: int, vector, alpha: float, max_new_tokens: int = 30,
             positions=None, **kw) -> str:
    """Generate with steering active (default: all positions incl. new tokens)."""
    with steering(model, layer, vector, alpha, positions):
        return model.generate(prompt, max_new_tokens=max_new_tokens, **kw)
