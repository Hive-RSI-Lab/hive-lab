"""Ablation: remove a direction (or a whole residual) and measure the damage.

Directional modes (u = unit direction), applied to the output of each block
in `layers` (lablib/jlens layer convention), at all positions by default:
- "project" (alias "zero"): h <- h - (h.u) u. The component along u is set
  to zero, i.e. zero-ablation of that one direction.
- "mean": h <- h - (h.u - mu) u, with mu the mean of h.u over reference
  prompts. The component is held at its average instead of zero, which
  avoids pushing the model off-distribution.

Whole-residual ablation (`ablate_residual`): zero or mean-replace the entire
residual at some positions. It is a blunt tool for "does this layer/position
matter at all"; the random-direction control does not apply to it.

The built-in control: `compare_to_random` ablates K random unit directions
in the same way. Removing ANY direction changes outputs a little; the real
direction only matters if its effect sits well outside that null.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager

import numpy as np

from . import stats


class _Ablate:
    def __init__(self, u, mu, positions):
        self.u, self.mu, self.positions = u, mu, positions
        self.lengths = None

    def __call__(self, h):
        u = self.u.to(h.device, h.dtype)
        coef = h @ u
        delta = (coef - (self.mu if self.mu is not None else 0.0)).unsqueeze(-1) * u
        if self.positions is None:
            return h - delta
        if h.shape[1] == 1:
            return None
        mask = _pos_mask(h, self.positions, self.lengths)
        return h - delta * mask


def _pos_mask(h, positions, lengths):
    import torch

    mask = torch.zeros(h.shape[0], h.shape[1], 1, dtype=h.dtype, device=h.device)
    lengths = lengths or [h.shape[1]] * h.shape[0]
    for b, n in enumerate(lengths):
        for p in positions:
            mask[b, p if p >= 0 else n + p] = 1
    return mask


def _unit(direction):
    import torch

    d = torch.as_tensor(direction, dtype=torch.float32).flatten()
    return d / d.norm()


def mean_projection(model, prompts, layer: int, direction) -> float:
    """Mean of h.u over all real tokens of `prompts` at `layer`."""
    import torch

    from .acts import get_acts

    a = get_acts(model, prompts, [layer], "all")[layer % model.n_layers]
    return float((torch.cat(a) @ _unit(direction)).mean())


@contextmanager
def ablation(model, layers, direction, mode: str = "project", ref_prompts=None,
             positions=None):
    """Context manager: ablate `direction` at each layer in `layers`.
    mode "mean" needs ref_prompts (to compute mu per layer)."""
    from .models import block_hook

    layers = [layers] if isinstance(layers, int) else list(layers)
    u = _unit(direction)
    hooks = []
    with ExitStack() as stack:
        for l in layers:
            if mode in ("project", "zero"):
                mu = None
            elif mode == "mean":
                if ref_prompts is None:
                    raise ValueError("mode='mean' needs ref_prompts")
                mu = mean_projection(model, ref_prompts, l, u)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            h = _Ablate(u, mu, positions)
            hooks.append(h)
            stack.enter_context(block_hook(model, l % model.n_layers, h))
        yield _Multi(hooks)


class _Multi:
    def __init__(self, hooks):
        self.hooks = hooks

    @property
    def lengths(self):
        return self.hooks[0].lengths if self.hooks else None

    @lengths.setter
    def lengths(self, v):
        for h in self.hooks:
            h.lengths = v


@contextmanager
def ablate_residual(model, layer: int, positions, mode: str = "zero", ref_prompts=None):
    """Replace the whole residual at `positions` with zeros ("zero") or with
    its mean over ref_prompts' real tokens ("mean")."""
    import torch

    from .acts import get_acts
    from .models import block_hook

    mean = None
    if mode == "mean":
        a = get_acts(model, ref_prompts, [layer], "all")[layer % model.n_layers]
        mean = torch.cat(a).mean(0)

    class H:
        lengths = None

        def __call__(self, h):
            if h.shape[1] == 1:
                return None
            m = _pos_mask(h, positions, self.lengths)
            rep = torch.zeros_like(h) if mean is None else mean.to(h.device, h.dtype).expand_as(h)
            return h * (1 - m) + rep * m

    hk = H()
    with block_hook(model, layer % model.n_layers, hk):
        yield hk


def kl_metric(clean_logits):
    """Metric factory: per-prompt KL(clean || ablated) of the next-token
    distribution. Bigger = more damage."""
    import torch

    lp_clean = torch.log_softmax(clean_logits.float(), -1)

    def f(logits):
        lp = torch.log_softmax(logits.float(), -1)
        return (lp_clean.exp() * (lp_clean - lp)).sum(-1).numpy()

    return f


def compare_to_random(model, prompts, layers, direction, mode: str = "project",
                      metric=None, k: int = 20, ref_prompts=None, positions=None,
                      seed: int = 0):
    """Ablate `direction` and K random unit directions the same way.

    metric(logits [n, vocab]) -> per-prompt values. Default: KL from the clean
    run (damage). If you pass e.g. steer.logit_diff_metric, the reported
    effect is the change from the clean value. Returns dict with effect,
    effect_ci, null_mean, null_sd, z, p, null, per_prompt."""
    import torch

    from .steer import last_logits

    clean = last_logits(model, prompts)
    if metric is None:
        m = kl_metric(clean)
        base = np.zeros(len(prompts))
    else:
        m = metric
        base = metric(clean)

    def run(d):
        with ablation(model, layers, d, mode, ref_prompts, positions) as ctx:
            return m(last_logits(model, prompts, ctx)) - base

    per = run(direction)
    g = torch.Generator().manual_seed(seed)
    d = _unit(direction).numel()
    null = [float(np.mean(run(torch.randn(d, generator=g)))) for _ in range(k)]
    obs, lo, hi = stats.bootstrap_ci(per, seed=seed)
    nm, sd = float(np.mean(null)), float(np.std(null, ddof=1)) if k > 1 else float("nan")
    return {"mode": mode, "layers": layers, "effect": obs, "effect_ci": (lo, hi),
            "null_mean": nm, "null_sd": sd,
            "z": (obs - nm) / sd if sd > 0 else float("nan"),
            "p": stats.empirical_p(obs - nm, np.array(null) - nm),
            "null": null, "per_prompt": per.tolist(), "k": k}


COMPARE_COLS = ["mode", "effect", "effect_ci", "null_mean", "null_sd", "z", "p"]
