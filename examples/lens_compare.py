"""Logit lens vs Jacobian lens on one prompt, layer by layer.

Run: lab run examples/lens_compare.py [--model gpt2] [--prompt "..."] [--target " Paris"]
Question: at which layer does the answer become readable, under each lens?
Control: the logit lens is the baseline; the last layer is the model's own output.
"""
import argparse

import lablib
from lablib import lens, report

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=lablib.DEFAULT)
ap.add_argument("--prompt", default="The Eiffel Tower is located in the city of")
ap.add_argument("--target", default=" Paris")
a = ap.parse_args()

m = lablib.load(a.model)
res = lens.compare(m, a.prompt, layers=None, k=3, target=a.target)
lens.print_compare(res)
report.run_dir("lens-compare").save(res, model=m, seed=None)
