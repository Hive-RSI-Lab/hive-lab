"""Linear probe for sentiment across layers, with controls.

Run: lab run examples/probe_sentiment.py [--model gpt2]
Controls in the table: majority-class accuracy, the same probe on shuffled
labels, grouped CV (adjectives never shared between train and test), and
control-task selectivity (Hewitt & Liang). A probe that beats majority and
shuffled shows the information is linearly readable - NOT that the model uses it.
"""
import argparse

import lablib
from lablib import acts, data, probe, report, stats

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=lablib.DEFAULT)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

m = lablib.load(a.model)
ds = data.sentiment()
layers = sorted(set([0, m.n_layers // 4, m.n_layers // 2, 3 * m.n_layers // 4, m.n_layers - 1]))
A = acts.get_acts(m, ds["prompts"], layers, position="last")
rows = probe.sweep_layers(A, ds["labels"], ds["groups"], seed=a.seed, control_task=True)
print(f"{len(ds['prompts'])} prompts, grouped by adjective; last-token residual")
stats.print_table(rows, probe.SWEEP_COLS)
report.run_dir("probe-sentiment").save({"rows": rows}, rows=rows, cols=probe.SWEEP_COLS,
                                       model=m, seed=a.seed)
