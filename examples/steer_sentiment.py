"""Steer with a sentiment difference-of-means vector vs random vectors.

Run: lab run examples/steer_sentiment.py [--model gpt2] [--k 10]
Metric: log p(" good") - log p(" bad") at the next token, on neutral prompts.
Control: K random vectors with the same norm at the same layer. The real
direction matters only if its effect sits far outside that null.
"""
import argparse

import lablib
from lablib import acts, data, probe, report, stats, steer

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
neutral = ["I went to see the new film and honestly it was",
           "We tried the restaurant on Main Street. The food was",
           "The weather over the weekend was",
           "My overall opinion of the book is that it is",
           "After the meeting, everyone agreed the plan was",
           "The hotel room was",
           "Her performance in the play was",
           "The service at the shop was"]
metric = steer.logit_diff_metric(m, " good", " bad")
print(f"layer {layer}; diff-of-means norm {dm['norm']:.1f}; "
      f"residual norm {steer.residual_norm(m, neutral, layer):.1f}; held-out dm acc {dm['acc']:.2f}")
rows = []
for alpha in (-1.0, 1.0):
    r = steer.compare_to_random(m, neutral, layer, dm["raw"], alpha, metric, k=a.k, seed=a.seed)
    rows.append(r)
stats.print_table(rows, steer.COMPARE_COLS)
print("effect = mean change in log p(good) - log p(bad); null = same-norm random vectors;"
      f" p floor with k={a.k} is {1/(a.k+1):.3f}")
for alpha in (-1.0, 1.0):
    print(f"alpha {alpha:+}:", repr(steer.generate(m, neutral[0], layer, dm["raw"], alpha,
                                                  max_new_tokens=8)))
report.run_dir("steer-sentiment").save({"rows": rows}, rows=rows, cols=steer.COMPARE_COLS,
                                       model=m, seed=a.seed)
