"""Small statistics helpers. Everything takes plain lists or numpy arrays."""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values, stat=np.mean, n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
                 groups=None):
    """Percentile bootstrap. Returns (point_estimate, low, high).

    With `groups` (one id per value) it resamples whole groups, not single values: a cluster
    bootstrap. Use it when values in a group are not independent, such as six prompts built from
    one word. Resampling single values there gives an interval that is too narrow."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        boots = np.array([stat(v[i]) for i in idx])
    else:
        g = np.asarray(groups)
        if len(g) != len(v):
            raise ValueError("groups must have one id per value")
        members = [np.flatnonzero(g == u) for u in np.unique(g)]
        pick = rng.integers(0, len(members), size=(n_boot, len(members)))
        boots = np.array([stat(v[np.concatenate([members[j] for j in row])]) for row in pick])
    a = (1 - ci) / 2
    return float(stat(v)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def permutation_test(a, b, n_perm: int = 10000, seed: int = 0, alternative: str = "two-sided"):
    """Unpaired test of mean(a) - mean(b) by permuting group labels.
    Returns (observed_diff, p_value). p uses the (1 + count) / (1 + n) rule."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = pooled[:len(a)].mean() - pooled[len(a):].mean()
        if alternative == "two-sided":
            count += abs(d) >= abs(obs) - 1e-12
        elif alternative == "greater":
            count += d >= obs - 1e-12
        else:
            count += d <= obs + 1e-12
    return float(obs), (count + 1) / (n_perm + 1)


def paired_test(a, b, n_perm: int = 10000, seed: int = 0):
    """Paired comparison of a vs b (same items). Sign-flip permutation test on
    the differences. Returns dict: mean_diff, ci (bootstrap 95%), p, dz
    (mean diff / sd of diffs), n."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(1)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    _, lo, hi = bootstrap_ci(d, seed=seed)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    return {"mean_diff": float(obs), "ci": (lo, hi), "p": float(p),
            "dz": float(obs / sd) if sd > 0 else float("nan"), "n": int(len(d))}


def cohens_d(a, b) -> float:
    """Cohen's d with pooled SD (ddof=1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the original order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def empirical_p(observed: float, null, two_sided: bool = True) -> float:
    """P-value of `observed` against samples from a null: (1 + #as-extreme) / (1 + n)."""
    null = np.asarray(null, float)
    if two_sided:
        c = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    else:
        c = np.sum(null >= observed - 1e-12)
    return float((c + 1) / (len(null) + 1))


def _fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.3g}" if abs(x) < 1e-3 and x != 0 else f"{x:.3f}"
    if isinstance(x, (tuple, list)) and len(x) == 2 and all(isinstance(v, float) for v in x):
        return f"[{x[0]:.3f}, {x[1]:.3f}]"
    return "-" if x is None else str(x)


def table_str(rows, cols) -> str:
    """Rows: list of lists (or dicts keyed by cols). Returns a markdown table."""
    rows = [[r.get(c) for c in cols] if isinstance(r, dict) else r for r in rows]
    cells = [[_fmt(x) for x in r] for r in rows]
    w = [max(len(str(c)), *(len(r[i]) for r in cells)) if cells else len(str(c))
         for i, c in enumerate(cols)]
    line = lambda xs: "| " + " | ".join(str(x).ljust(w[i]) for i, x in enumerate(xs)) + " |"
    out = [line(cols), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += [line(r) for r in cells]
    return "\n".join(out)


def print_table(rows, cols) -> None:
    print(table_str(rows, cols))
