"""Activation patching: where does "France" -> " Paris" flow?

Run: lab run examples/patch_capitals.py [--model gpt2]
Clean: "The capital of France is" (answer " Paris"); corrupt: same with
Italy (" Rome"). Each cell patches one (layer, position) from clean into
corrupt. Metric: log p(" Paris") - log p(" Rome").
Expect: the country token carries it in early/middle layers, the last token
in later layers. One prompt pair is an illustration, not a result.
"""
import argparse

import lablib
from lablib import patch, report, steer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=lablib.DEFAULT)
ap.add_argument("--clean", default="The capital of France is")
ap.add_argument("--corrupt", default="The capital of Italy is")
ap.add_argument("--answers", default=" Paris, Rome")
a = ap.parse_args()

m = lablib.load(a.model)
good, bad = [s if s.startswith(" ") else " " + s for s in
             (x.strip() for x in a.answers.split(","))]
metric = steer.logit_diff_metric(m, good, bad)
step = max(1, m.n_layers // 12)
layers = sorted(set(list(range(0, m.n_layers, step)) + [m.n_layers - 1]))
res = patch.patch_scan(m, a.clean, a.corrupt, metric, layers=layers)
patch.print_scan(res)
report.run_dir("patch-capitals").save(res, model=m)
