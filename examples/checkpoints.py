"""Next-word scores of Pythia-70m at two training checkpoints and with random weights.

Run: lab run examples/checkpoints.py
"""
import time

import torch

import lablib

t0 = time.time()
NAME, PROMPT = "pythia-70m", "The capital of France is"
variants = {
    "step1000": dict(revision="step1000"),
    "step143000": dict(revision="step143000"),
    "random seed 0": dict(random_seed=0),
    "random seed 0 again": dict(random_seed=0),
}
scores = {}
for label, kw in variants.items():
    m = lablib.load(NAME, device="cpu", verbose=False, **kw)
    enc = m.tok(PROMPT, return_tensors="pt")
    with torch.no_grad():
        lg = m.hf(**enc).logits[0, -1].float()
    scores[label] = lg
    top = lg.topk(3).indices.tolist()
    print(f"{label:22s} top3 {[m.tok.decode(i) for i in top]}  logit sum {lg.sum():.4f}  "
          f"first5 {[round(x, 3) for x in lg[:5].tolist()]}")
    del m
labs = list(scores)
for i in range(3):
    for j in range(i + 1, 3):
        d = (scores[labs[i]] - scores[labs[j]]).abs().max().item()
        print(f"max |diff| {labs[i]} vs {labs[j]}: {d:.4f}")
same = torch.equal(scores["random seed 0"], scores["random seed 0 again"])
print("same seed twice identical:", same)
assert same
assert all(not torch.allclose(scores[a], scores[b]) for a, b in [(labs[0], labs[1]), (labs[0], labs[2]), (labs[1], labs[2])])
print(f"OK three differ, same seed identical; {time.time() - t0:.0f} s")
