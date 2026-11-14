"""Ablate a sentiment direction vs random directions.

Run: lab run examples/ablate_sentiment.py [--model gpt2] [--k 10]
Metric: log p(" good") - log p(" bad") after sentiment-laden prompts; the
effect is the change from the clean run (positive prompts should drop,
negative prompts rise). Control: the same ablation of K random unit directions.
"""
import argparse

import numpy as np

import lablib
from lablib import ablate, acts, data, probe, report, stats, steer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=lablib.DEFAULT)
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

m = lablib.load(a.model)
layer = a.layer % m.n_layers if a.layer is not None else m.n_layers // 2   # negative counts from the end
ds = data.sentiment()
A = acts.get_acts(m, ds["prompts"], [layer], position="last")[layer]
dm = probe.diff_means(A, ds["labels"], ds["groups"], seed=a.seed)
pos = ["The movie was wonderful. In one word, it was",
       "What a lovely, delightful day. Overall it was",
       "The food was excellent and the staff were charming. It was"]
neg = ["The movie was awful. In one word, it was",
       "What a miserable, dreadful day. Overall it was",
       "The food was terrible and the staff were rude. It was"]
metric = steer.logit_diff_metric(m, " good", " bad")
rows = []
for name, prompts in (("positive", pos), ("negative", neg)):
    for mode in ("project", "mean"):
        r = ablate.compare_to_random(m, prompts, [layer], dm["direction"], mode=mode,
                                     metric=metric, k=a.k, ref_prompts=ds["prompts"],
                                     seed=a.seed)
        r["prompts"] = name
        rows.append(r)
cols = ["prompts"] + ablate.COMPARE_COLS
print(f"layer {layer} (all positions); effect = change in log p(good) - log p(bad)")
stats.print_table(rows, cols)
print(f"p floor with k={a.k} is {1/(a.k+1):.3f}; few prompts, so treat this as a demo")
report.run_dir("ablate-sentiment").save({"rows": rows}, rows=rows, cols=cols, model=m,
                                        seed=a.seed)
