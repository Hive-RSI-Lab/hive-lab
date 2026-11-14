"""Activation patching (causal tracing) over layers x positions.

Run the CORRUPT prompt, but replace the output of block `layer` at token
`position` with its value from the CLEAN run. If the metric recovers, that
(layer, position) carries information the clean answer needs.

    recovery = (patched - corrupt) / (clean - corrupt)

1 = fully restores the clean behaviour, 0 = no effect. Clean and corrupt
prompts must tokenise to the same length (positions are compared one to
one); `check_pair` tells you if they do not.

Controls/caveats: patching one site at a time misses effects spread over many
sites; recovery is noisy when clean - corrupt is small (we report the raw
gap so you can see it); average over several prompt pairs before
concluding anything.
"""

from __future__ import annotations

import numpy as np


def check_pair(model, clean: str, corrupt: str) -> int:
    a, b = model.encode([clean])["lengths"][0], model.encode([corrupt])["lengths"][0]
    if a != b:
        raise ValueError(f"clean has {a} tokens, corrupt has {b}: patching needs equal "
                         f"lengths. clean={model.tokens(clean)} corrupt={model.tokens(corrupt)}")
    return a


def patch_scan(model, clean: str, corrupt: str, metric, layers=None, positions=None):
    """Patch each (layer, position) from clean into corrupt.

    metric(logits [n, vocab]) -> numpy [n], e.g.
    steer.logit_diff_metric(model, " Paris", " Rome"). Returns dict with
    recovery [len(layers), len(positions)], patched (raw metric), clean,
    corrupt, layers, positions, tokens."""
    import torch

    from .acts import get_acts
    from .models import block_hook

    n = check_pair(model, clean, corrupt)
    layers = list(range(model.n_layers)) if layers is None else [l % model.n_layers for l in layers]
    positions = list(range(n)) if positions is None else [p % n for p in positions]
    clean_acts = get_acts(model, [clean], layers, "all", cache=False)
    enc = model.encode([corrupt] * len(positions))
    with torch.no_grad():
        m_clean = float(metric(model.last_logits([clean]))[0])
        m_corr = float(metric(model.last_logits([corrupt]))[0])
        out = np.zeros((len(layers), len(positions)))
        for i, l in enumerate(layers):
            src = clean_acts[l][0]

            def fn(h, src=src):
                if h.shape[1] != n:
                    return None
                h = h.clone()
                for b, p in enumerate(positions):
                    h[b, p] = src[p].to(h.device, h.dtype)
                return h

            with block_hook(model, l, fn):
                lg = model.hf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                              use_cache=False).logits[:, -1].float().cpu()
            out[i] = metric(lg)
    gap = m_clean - m_corr
    rec = (out - m_corr) / gap if abs(gap) > 1e-6 else np.full_like(out, np.nan)
    return {"clean_prompt": clean, "corrupt_prompt": corrupt, "clean": m_clean,
            "corrupt": m_corr, "gap": gap, "layers": layers, "positions": positions,
            "tokens": model.tokens(clean), "corrupt_tokens": model.tokens(corrupt),
            "patched": out.tolist(), "recovery": rec.tolist()}


def print_scan(res, max_tok: int = 16) -> None:
    """Recovery table: rows = layers, columns = token positions."""
    from .stats import print_table

    toks = [res["tokens"][p] for p in res["positions"]]
    ctoks = [res["corrupt_tokens"][p] for p in res["positions"]]
    cols = ["layer"] + [(t if t == c else f"{t}/{c}").replace("\n", "\\n")[:max_tok]
                        for t, c in zip(toks, ctoks)]
    rows = [[l] + [round(v, 2) for v in r] for l, r in zip(res["layers"], res["recovery"])]
    print(f"clean metric {res['clean']:.2f}, corrupt {res['corrupt']:.2f}, gap {res['gap']:.2f}")
    print_table(rows, cols)
    print("cells: recovery = (patched - corrupt) / (clean - corrupt); "
          "column 'a/b' = clean/corrupt token")
