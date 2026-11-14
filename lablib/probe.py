"""Linear probes with controls reported by default.

Every probe result carries, next to the test accuracy:
- majority: accuracy of always guessing the most common training label;
- shuffled: accuracy of the same probe trained on shuffled training labels
  (mean over n_shuffle refits) - what "fitting noise" scores here;
- optionally control_task (Hewitt & Liang 2019): each group (e.g. word type)
  gets a random label. selectivity = seen_acc - control_acc, where both probes
  are fitted and scored on one split that puts every group on both sides
  (see control_scores).

The train/test split for `acc` is GROUPED: all prompts sharing a group id (same
template, same word, same entity) land on the same side, so near-duplicates
can't leak from train to test. Always pass groups when prompts are built
from templates. `acc_ci` resamples whole test groups (a cluster bootstrap),
because prompts that share a group are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats


def _np(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _clf(C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000, random_state=seed))


def group_split(y, groups, test_size: float = 0.3, seed: int = 0):
    """Indices (train, test) with whole groups on one side. Retries seeds so
    both sides contain every class when possible."""
    from sklearn.model_selection import GroupShuffleSplit

    y = np.asarray(y)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    for s in range(seed, seed + 50):
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(
            y, y, groups))
        if len(set(y[tr])) == len(set(y)) and len(set(y[te])) == len(set(y)):
            return tr, te
    return tr, te


@dataclass
class ProbeResult:
    acc: float
    acc_ci: tuple
    majority: float
    shuffled: float
    shuffled_sd: float
    cv_acc: float | None
    n_train: int
    n_test: int
    seen_acc: float | None = None
    control_acc: float | None = None
    selectivity: float | None = None
    correct: list = field(default_factory=list, repr=False)
    clf: object = field(default=None, repr=False)

    def row(self):
        return {"acc": self.acc, "95% CI": self.acc_ci, "majority": self.majority,
                "shuffled": self.shuffled, "cv_acc": self.cv_acc, "seen_acc": self.seen_acc,
                "control": self.control_acc, "selectivity": self.selectivity,
                "n_test": self.n_test}

    def __str__(self):
        return stats.table_str([self.row()], SWEEP_COLS[1:])


def fit_probe(X, y, groups=None, test_size: float = 0.3, seed: int = 0, C: float = 1.0,
              n_shuffle: int = 10, cv_folds: int = 5, control_task: bool = False,
              n_boot: int = 1000) -> ProbeResult:
    """Train a logistic-regression probe on X [n, d] -> y and report controls.

    acc is accuracy on test groups the probe never saw. acc_ci is a bootstrap
    95% interval that resamples whole test groups; with few test groups it is
    wide, and that is the honest width. cv_acc is grouped k-fold accuracy
    (None if too few groups). With control_task=True the result also carries
    seen_acc, control_acc and selectivity from control_scores()."""
    from sklearn.model_selection import GroupKFold

    X, y = _np(X), np.asarray(y)
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    tr, te = group_split(y, g, test_size, seed)
    clf = _clf(C, seed).fit(X[tr], y[tr])
    correct = (clf.predict(X[te]) == y[te]).astype(float)
    acc, lo, hi = stats.bootstrap_ci(correct, n_boot=n_boot, seed=seed, groups=g[te])
    vals, counts = np.unique(y[tr], return_counts=True)
    majority = float(np.mean(y[te] == vals[np.argmax(counts)]))
    rng = np.random.default_rng(seed)
    sh = []
    for _ in range(n_shuffle):
        ys = rng.permutation(y[tr])
        if len(set(ys)) < 2:
            continue
        sh.append(float(np.mean(_clf(C, seed).fit(X[tr], ys).predict(X[te]) == y[te])))
    cv = None
    n_groups = len(set(g.tolist()))
    if n_groups >= cv_folds:
        accs = []
        for a, b in GroupKFold(cv_folds).split(X, y, g):
            if len(set(y[a])) < 2:
                continue
            accs.append(float(np.mean(_clf(C, seed).fit(X[a], y[a]).predict(X[b]) == y[b])))
        cv = float(np.mean(accs)) if accs else None
    res = ProbeResult(acc=acc, acc_ci=(lo, hi), majority=majority,
                      shuffled=float(np.mean(sh)) if sh else float("nan"),
                      shuffled_sd=float(np.std(sh)) if sh else float("nan"), cv_acc=cv,
                      n_train=len(tr), n_test=len(te), correct=correct.tolist(), clf=clf)
    if control_task:
        cs = control_scores(X, y, g, test_size, seed, C)
        res.seen_acc, res.control_acc = cs["seen_acc"], cs["control_acc"]
        res.selectivity = cs["selectivity"]
    return res


def seen_split(groups, test_size: float = 0.3, seed: int = 0):
    """Indices (train, test) split by ITEM inside each group: every group with two or more items
    has at least one on each side. Groups with one item go to train. This is the opposite of
    group_split, and it is what a control task needs: the probe must have seen each test group."""
    g = np.asarray(groups)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for u in np.unique(g):
        idx = rng.permutation(np.flatnonzero(g == u))
        k = min(len(idx) - 1, max(1, int(round(len(idx) * test_size)))) if len(idx) > 1 else 0
        te.extend(idx[:k].tolist())
        tr.extend(idx[k:].tolist())
    return np.array(sorted(tr), dtype=int), np.array(sorted(te), dtype=int)


def control_scores(X, y, groups, test_size: float = 0.3, seed: int = 0, C: float = 1.0) -> dict:
    """Hewitt & Liang (2019) selectivity, with both probes on ONE split.

    The split is seen_split: by item, so the probe has seen every test group in training. Two
    probes are fitted on the same train items and scored on the same test items:
      seen_acc    - the real labels;
      control_acc - the control task: each group gets one random label, drawn with the real
                    class proportions. Only memorising group identity can score here.
    selectivity = seen_acc - control_acc. A value near 0 means that, with groups seen in
    training, the probe does no better on the real labels than on made-up ones, so seen_acc is
    no evidence of a feature. It does not tell you about `acc` on unseen groups; read that next
    to it.

    Returns a dict: seen_acc, control_acc, selectivity, n_train, n_test. The three scores are
    NaN when no group has two items (pass real groups) or a label set has one class."""
    X, y, g = _np(X), np.asarray(y), np.asarray(groups)
    nan = float("nan")
    tr, te = seen_split(g, test_size, seed)
    out = {"seen_acc": nan, "control_acc": nan, "selectivity": nan,
           "n_train": int(len(tr)), "n_test": int(len(te))}
    if len(te) == 0 or len(set(y[tr].tolist())) < 2:
        return out
    ug = np.unique(g)
    yc = None
    for s in range(seed + 1, seed + 21):  # redraw if every group got the same label
        rng = np.random.default_rng(s)
        lab = dict(zip(ug.tolist(), rng.choice(y, size=len(ug), replace=True).tolist()))
        cand = np.array([lab[x] for x in g.tolist()])
        if len(set(cand[tr].tolist())) > 1:
            yc = cand
            break
    out["seen_acc"] = float(np.mean(_clf(C, seed).fit(X[tr], y[tr]).predict(X[te]) == y[te]))
    if yc is None:
        return out
    out["control_acc"] = float(np.mean(_clf(C, seed).fit(X[tr], yc[tr]).predict(X[te]) == yc[te]))
    out["selectivity"] = out["seen_acc"] - out["control_acc"]
    return out


def control_task_acc(X, y, groups, test_size=0.3, seed=0, C=1.0) -> float:
    """Control-task accuracy alone: control_scores(...)["control_acc"]."""
    return control_scores(X, y, groups, test_size, seed, C)["control_acc"]


def diff_means(X, y, groups=None, test_size: float = 0.3, seed: int = 0):
    """Difference-of-means direction (mean of class 1 minus class 0) fitted on
    the train split. Returns dict: direction (unit, torch float32), raw (the
    unnormalised difference, torch), norm, acc (test accuracy of thresholding
    the projection at the midpoint, on unseen groups), ci (bootstrap over test groups)."""
    import torch

    X, y = _np(X), np.asarray(y)
    classes = np.unique(y)
    assert len(classes) == 2, "diff_means needs exactly two classes"
    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    tr, te = group_split(y, g, test_size, seed)
    m1, m0 = X[tr][y[tr] == classes[1]].mean(0), X[tr][y[tr] == classes[0]].mean(0)
    raw = m1 - m0
    u = raw / np.linalg.norm(raw)
    thr = ((m1 + m0) / 2) @ u
    pred = np.where(X[te] @ u > thr, classes[1], classes[0])
    correct = (pred == y[te]).astype(float)
    acc, lo, hi = stats.bootstrap_ci(correct, seed=seed, groups=g[te])
    return {"direction": torch.tensor(u, dtype=torch.float32),
            "raw": torch.tensor(raw, dtype=torch.float32),
            "norm": float(np.linalg.norm(raw)), "acc": acc, "ci": (lo, hi)}


def sweep_layers(acts: dict, y, groups=None, seed: int = 0, n_shuffle: int = 5,
                 control_task: bool = False, n_boot: int = 1000):
    """Probe every layer in `acts` ({layer: [n, d]}). Returns a list of row
    dicts with the keys in SWEEP_COLS. seen_acc, control and selectivity are
    None unless control_task=True.
    Print with lablib.stats.print_table(rows, SWEEP_COLS)."""
    rows = []
    for l in sorted(acts):
        r = fit_probe(acts[l], y, groups, seed=seed, n_shuffle=n_shuffle,
                      control_task=control_task, n_boot=n_boot)
        rows.append({"layer": l, **r.row()})
    return rows


SWEEP_COLS = ["layer", "acc", "95% CI", "majority", "shuffled", "cv_acc", "seen_acc", "control",
              "selectivity", "n_test"]
