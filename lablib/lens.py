"""Logit lens and Jacobian lens, side by side.

Logit lens: unembed(final_norm(h_L)).
Jacobian lens (jlens): unembed(final_norm(J_L @ h_L)), where J_L is the
average Jacobian of the final residual with respect to the layer-L residual,
fitted on generic text. Layer indexing is jlens's: L = output of block L.
The last layer (n_layers-1) is the model's own output for both lenses.
"""

from __future__ import annotations

import os
from pathlib import Path

from .models import KNOWN, LENS_REPO, LENS_REVISION


def _lens_cache_dir() -> Path:
    d = Path(os.environ.get("LAB_CACHE", Path.home() / ".cache" / "lab")) / "lenses"
    d.mkdir(parents=True, exist_ok=True)
    return d


_FALLBACK_TEXT = [
    "The history of the city goes back to a small fishing village on the river. Over the "
    "centuries it grew into a market town, then an industrial centre with mills, docks and "
    "a railway station that linked it to the capital and to the coast.",
    "In mathematics, a prime number is a natural number greater than one that is not a "
    "product of two smaller natural numbers. Primes are central in number theory because "
    "every natural number greater than one is a product of primes.",
    "The band released its first album in the spring, recorded in a converted barn over three "
    "weeks. Critics praised the songwriting but found the production thin, and the record "
    "sold modestly until a single was used in a television advertisement.",
    "Photosynthesis is the process by which green plants and some other organisms use "
    "sunlight to synthesize foods from carbon dioxide and water. It generally involves the "
    "green pigment chlorophyll and produces oxygen as a by-product.",
]


def _fit_prompts(n: int) -> tuple[list[str], str]:
    try:
        from jlens.examples import load_wikitext_prompts

        p = load_wikitext_prompts(n_prompts=n)
        if len(p) == n:
            return p, "wikitext-103"
    except Exception as e:  # noqa: BLE001
        print(f"[lab] wikitext unavailable ({type(e).__name__}); using built-in text")
    return (_FALLBACK_TEXT * (n // len(_FALLBACK_TEXT) + 1))[:n], "built-in paragraphs"


def get_lens(model, n_fit: int = 16, allow_fit: bool = True):
    """Return (JacobianLens, description). Uses the published neuronpedia lens
    when one is known for the model; otherwise fits one on n_fit prompts and
    caches it (description then says SMALL-SAMPLE FIT)."""
    import jlens
    from huggingface_hub import hf_hub_download

    spec = KNOWN.get(model.name, {})
    if spec.get("lens"):
        path = hf_hub_download(LENS_REPO, spec["lens"], revision=LENS_REVISION)
        lens = jlens.JacobianLens.load(path)
        return lens, f"published lens {LENS_REPO}/{spec['lens']} (n_prompts={lens.n_prompts})"
    cached = _lens_cache_dir() / f"{model.name.replace('/', '--')}-n{n_fit}.pt"
    if cached.exists():
        lens = jlens.JacobianLens.load(str(cached))
        return lens, f"SMALL-SAMPLE FIT (n={lens.n_prompts} prompts, cached {cached})"
    if not allow_fit:
        raise FileNotFoundError(f"no published or cached lens for {model.name}")
    prompts, src = _fit_prompts(n_fit)
    print(f"[lab] no published lens for {model.name}: fitting on {n_fit} prompts from {src} "
          "(SMALL-SAMPLE FIT; this can take minutes on CPU)", flush=True)
    lens = jlens.fit(model.lm, prompts, dim_batch=32, max_seq_len=128)
    lens.save(str(cached))
    return lens, f"SMALL-SAMPLE FIT (n={lens.n_prompts} prompts from {src}, cached {cached})"


def residuals(model, prompt: str, layers):
    """{layer: [seq, d_model] float32} for one prompt (jlens tokenisation)."""
    import torch
    from jlens import ActivationRecorder

    ids = model.lm.encode(prompt)
    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as rec:
        model.lm.forward(ids)
        acts = {l: rec.activations[l][0].detach().float() for l in layers}
    return acts, ids[0].tolist()


def compare(model, prompt: str, layers=None, position: int = -1, k: int = 5,
            target: str | None = None, lens=None, lens_desc: str | None = None):
    """Read the residual at `position` through both lenses at each layer.

    Returns dict with rows: one per layer with logit-lens and J-lens top-k
    tokens, and (if target given) the target token's rank (0 = top) and
    probability under each lens. Also returns the lens description."""
    import torch

    if lens is None:
        lens, lens_desc = get_lens(model)
    if layers is None:
        step = max(1, model.n_layers // 8)
        layers = sorted(set(list(range(0, model.n_layers, step)) + [model.n_layers - 1]))
    layers = [l % model.n_layers for l in layers]
    acts, ids = residuals(model, prompt, layers)
    tid = model.token_id(target) if target is not None else None
    rows = []
    with torch.no_grad():
        for l in layers:
            h = acts[l][position]
            ll = model.lm.unembed(h).float().cpu()
            if l in lens.jacobians:
                jl = model.lm.unembed(lens.transport(h, l)).float().cpu()
            elif l == model.n_layers - 1:
                jl = ll  # final layer: J = identity
            else:
                jl = None
            row = {"layer": l,
                   "logit_lens_top": [model.tok.decode([int(t)]) for t in ll.topk(k).indices],
                   "jlens_top": None if jl is None else
                   [model.tok.decode([int(t)]) for t in jl.topk(k).indices]}
            if tid is not None:
                for nm, lg in (("logit", ll), ("jlens", jl)):
                    if lg is None:
                        row[f"{nm}_rank"] = row[f"{nm}_prob"] = None
                        continue
                    row[f"{nm}_rank"] = int((lg > lg[tid]).sum())
                    row[f"{nm}_prob"] = float(torch.softmax(lg, -1)[tid])
            rows.append(row)
    return {"prompt": prompt, "position": position,
            "token_at_position": model.tok.decode([ids[position]]),
            "target": target, "lens": lens_desc, "rows": rows}


def print_compare(res, width: int = 34) -> None:
    from .stats import print_table

    print(f"lens: {res['lens']}")
    print(f"prompt: {res['prompt']!r}; reading position {res['position']} "
          f"(token {res['token_at_position']!r})")
    cols = ["layer", "logit lens top", "J-lens top"]
    if res["target"] is not None:
        cols += ["logit rank", "J rank"]
    rows = []
    for r in res["rows"]:
        fmt = lambda xs: "-" if xs is None else "|".join(x.replace("\n", "\\n") for x in xs)[:width]
        row = [r["layer"], fmt(r["logit_lens_top"]), fmt(r["jlens_top"])]
        if res["target"] is not None:
            row += [r["logit_rank"], r["jlens_rank"]]
        rows.append(row)
    print_table(rows, cols)
    if res["target"] is not None:
        print(f"(rank of target {res['target']!r}: 0 = top token)")
