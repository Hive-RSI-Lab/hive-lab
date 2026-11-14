"""Tiny contrast datasets, generated in code (no downloads).

Each dataset function returns a dict with:
  prompts  list[str]
  labels   list[int]   (1 / 0)
  groups   list        (group id: prompts sharing it must stay on one side of
                        a train/test split - pass it to probe.fit_probe)
  meta     dict        (what the classes are, token pairs for logit metrics)

These are small on purpose: enough for a quick probe or steering vector, far
too small to support claims about "the model's concept of X".
"""

from __future__ import annotations

import itertools
import random

_POS = ["wonderful", "excellent", "delightful", "brilliant", "lovely", "fantastic",
        "superb", "charming", "great", "pleasant", "amazing", "beautiful"]
_NEG = ["terrible", "awful", "dreadful", "horrible", "boring", "disappointing",
        "miserable", "poor", "bad", "unpleasant", "painful", "ugly"]
_SENT_TEMPLATES = [
    "The movie was {adj}.",
    "I thought the food was {adj}.",
    "Our trip to the coast was {adj}.",
    "Honestly, the concert was {adj}.",
    "The new book is {adj}.",
    "My day at work was {adj}.",
]


def sentiment(n_templates: int | None = None):
    """Positive (1) vs negative (0) sentences: templates x adjectives.
    Grouped by adjective, so a probe is tested on unseen adjectives."""
    tpl = _SENT_TEMPLATES[:n_templates] if n_templates else _SENT_TEMPLATES
    prompts, labels, groups = [], [], []
    for lab, words in ((1, _POS), (0, _NEG)):
        for w in words:
            for t in tpl:
                prompts.append(t.format(adj=w))
                labels.append(lab)
                groups.append(w)
    return {"prompts": prompts, "labels": labels, "groups": groups,
            "meta": {"classes": {1: "positive", 0: "negative"},
                     "pos_words": list(_POS), "neg_words": list(_NEG)}}


CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Japan", "Tokyo"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Canada", "Ottawa"),
    ("Greece", "Athens"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Kenya", "Nairobi"),
    ("Peru", "Lima"), ("Thailand", "Bangkok"), ("Cuba", "Havana"), ("Hungary", "Budapest"),
]


def true_false(seed: int = 0):
    """True (1) / false (0) statements "The capital of X is Y." Each country
    appears once true and once with a wrong capital. Grouped by country."""
    rng = random.Random(seed)
    caps = [c for _, c in CAPITALS]
    prompts, labels, groups = [], [], []
    for country, cap in CAPITALS:
        wrong = rng.choice([c for c in caps if c != cap])
        for lab, city in ((1, cap), (0, wrong)):
            prompts.append(f"The capital of {country} is {city}.")
            labels.append(lab)
            groups.append(country)
    return {"prompts": prompts, "labels": labels, "groups": groups,
            "meta": {"classes": {1: "true", 0: "false"}}}


def capital_prompts():
    """Completion prompts "The capital of X is" with the answer token " Y".
    Returns list of (prompt, answer) - for lens and patching."""
    return [(f"The capital of {c} is", f" {cap}") for c, cap in CAPITALS]


def pairs_from_template(template: str, slot_a, slot_b, **fixed):
    """Build contrast pairs from one template.

    template: a str with {x} for the varying slot (and any other {names}
    filled from `fixed`, where each value may be a str or a list to expand).
    slot_a, slot_b: equal-length lists; pair i = (template with x=slot_a[i],
    template with x=slot_b[i]).
    Returns list of (prompt_a, prompt_b), expanded over list-valued fixed slots.

    Example: pairs_from_template("The {x} cat sat on the", ["black"], ["white"]).
    """
    assert len(slot_a) == len(slot_b), "slot lists must have equal length"
    keys = list(fixed)
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in fixed.values()]
    out = []
    for combo in itertools.product(*vals) if keys else [()]:
        kw = dict(zip(keys, combo))
        for a, b in zip(slot_a, slot_b):
            out.append((template.format(x=a, **kw), template.format(x=b, **kw)))
    return out
