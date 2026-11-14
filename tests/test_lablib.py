"""lablib tests on gpt2 (small, fast). Tests marked `quick` run in `lab check`.

Run all: lab py -m pytest tests/test_lablib.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lablib import stats  # noqa: E402

quick = pytest.mark.quick


@pytest.fixture(scope="module")
def m():
    import lablib

    return lablib.load("gpt2", device="cpu", verbose=False)


# ---------------------------------------------------------------- stats
@quick
def test_stats_known_values():
    v = [1.0, 2.0, 3.0, 4.0]
    est, lo, hi = stats.bootstrap_ci(v, seed=0)
    assert est == 2.5 and 1.0 <= lo < 2.5 < hi <= 4.0
    # Cohen's d: means 2 and 4, pooled sd 1 -> d = -2
    assert stats.cohens_d([1, 2, 3], [3, 4, 5]) == pytest.approx(-2.0)
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.empirical_p(5.0, [0, 1, -1, 2]) == pytest.approx(1 / 5)
    assert stats.empirical_p(0.5, [0, 1, -1, 2]) == pytest.approx(4 / 5)
    obs, p = stats.permutation_test([10, 11, 12, 13], [0, 1, 2, 3], n_perm=2000)
    assert obs == 10 and p < 0.05  # exact p = 2/70
    obs, p = stats.permutation_test([1, 2, 3], [1, 2, 3], n_perm=500)
    assert obs == 0 and p == 1.0
    r = stats.paired_test([2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8])
    assert r["mean_diff"] == 1 and r["p"] < 0.02  # all 8 diffs positive: exact p = 2/256


@quick
def test_bootstrap_resamples_whole_groups():
    # 10 groups of 6 identical values: there are 10 independent numbers here, not 60
    v = np.repeat([0, 1, 0, 1, 1, 0, 1, 1, 0, 1], 6).astype(float)
    g = np.repeat(np.arange(10), 6)
    est, lo, hi = stats.bootstrap_ci(v, seed=0)
    est_g, lo_g, hi_g = stats.bootstrap_ci(v, seed=0, groups=g)
    assert est == est_g == pytest.approx(0.6)
    assert (hi_g - lo_g) > 1.5 * (hi - lo)  # the item-level interval is too narrow
    # every resample is a set of whole groups, so its mean is a multiple of 1/10
    _, lo1, hi1 = stats.bootstrap_ci(v, seed=1, groups=g, n_boot=200)
    assert round(lo1 * 10, 6) == round(lo1 * 10) and round(hi1 * 10, 6) == round(hi1 * 10)
    # one group per value is the plain bootstrap in distribution
    _, lo_s, hi_s = stats.bootstrap_ci(v, seed=0, groups=np.arange(len(v)))
    assert abs((hi_s - lo_s) - (hi - lo)) < 0.08
    with pytest.raises(ValueError):
        stats.bootstrap_ci(v, groups=g[:-1])


@quick
def test_table_str():
    s = stats.table_str([{"a": 1, "b": 0.5, "c": (0.1, 0.2)}], ["a", "b", "c"])
    assert "| 1 " in s and "0.500" in s and "[0.100, 0.200]" in s


# ---------------------------------------------------------------- probe
@quick
def test_probe_separable_vs_shuffled():
    from lablib import probe

    rng = np.random.default_rng(0)
    n, d = 200, 20
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, d))
    X[:, 0] += 4 * y  # class info along one axis
    groups = np.arange(n) // 4
    r = probe.fit_probe(X, y, groups, n_shuffle=10, control_task=True)
    assert r.acc > 0.95
    assert abs(r.majority - 0.5) < 0.15
    assert abs(r.shuffled - 0.5) < 0.15  # shuffled-label control sits at chance
    assert r.cv_acc > 0.95
    noise = probe.fit_probe(rng.normal(size=(n, d)), y, groups, n_shuffle=5)
    assert noise.acc < 0.75


@quick
def test_group_split_keeps_groups_apart():
    from lablib import probe

    y = np.repeat([0, 1], 50)
    g = np.arange(100) // 5
    tr, te = probe.group_split(y, g)
    assert not set(g[tr]) & set(g[te])


@quick
def test_seen_split_puts_each_group_on_both_sides():
    from lablib import probe

    g = np.array([0] * 6 + [1] * 2 + [2] * 1 + [3] * 10)
    tr, te = probe.seen_split(g, test_size=0.3, seed=0)
    assert sorted(tr.tolist() + te.tolist()) == list(range(len(g)))
    assert not set(tr.tolist()) & set(te.tolist())
    for u in (0, 1, 3):  # two or more items: on both sides
        assert u in g[tr] and u in g[te]
    assert 2 in g[tr] and 2 not in g[te]  # a single item stays in train
    assert (g[te] == 3).sum() == 3 and (g[te] == 0).sum() == 2


@quick
def test_control_scores_selectivity():
    from lablib import probe

    rng = np.random.default_rng(0)
    n_groups, per, d = 40, 6, 24
    g = np.repeat(np.arange(n_groups), per)
    y_group = rng.permutation(np.repeat([0, 1], n_groups // 2))
    y = y_group[g]

    # (a) a real feature: one axis carries the label, nothing identifies the group
    Xa = rng.normal(size=(len(g), d))
    Xa[:, 0] += 6 * y  # three standard deviations each side of the boundary
    a = probe.control_scores(Xa, y, g)
    assert a["seen_acc"] > 0.95
    assert abs(a["control_acc"] - 0.5) < 0.2  # made-up labels cannot be read off
    assert a["selectivity"] == pytest.approx(a["seen_acc"] - a["control_acc"])
    assert a["selectivity"] > 0.3
    assert a["n_train"] + a["n_test"] == len(g) and a["n_test"] == n_groups * 2

    # (b) only group identity: each group is its own cluster, the label is a property of the group
    centres = rng.normal(size=(n_groups, d)) * 4
    Xb = centres[g] + 0.1 * rng.normal(size=(len(g), d))
    b = probe.control_scores(Xb, y, g)
    assert b["seen_acc"] > 0.8 and b["control_acc"] > 0.8  # both are memorised
    assert abs(b["selectivity"]) < 0.2
    # ... and the grouped split shows it: on unseen groups the probe is far from perfect
    assert probe.fit_probe(Xb, y, g, n_shuffle=2, n_boot=50).acc < 0.85
    assert probe.control_task_acc(Xb, y, g) == b["control_acc"]

    # no group with two items: nothing to score
    c = probe.control_scores(Xa, y, np.arange(len(g)))
    assert np.isnan(c["seen_acc"]) and np.isnan(c["selectivity"]) and c["n_test"] == 0

    r = probe.fit_probe(Xa, y, g, n_shuffle=2, control_task=True, n_boot=50)
    assert r.seen_acc == a["seen_acc"] and r.control_acc == a["control_acc"]
    assert set(probe.SWEEP_COLS) >= {"seen_acc", "control", "selectivity"}
    assert r.row()["control"] == a["control_acc"]
    assert probe.fit_probe(Xa, y, g, n_shuffle=2, n_boot=50).selectivity is None  # off by default


@quick
def test_probe_ci_is_over_groups():
    from lablib import probe

    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(30), 6)
    y = np.repeat(rng.permutation(np.repeat([0, 1], 15)), 6)
    X = rng.normal(size=(len(g), 8))
    X[:, 0] += 1.0 * y  # weak signal, so accuracy is well inside (0, 1)
    r = probe.fit_probe(X, y, g, n_shuffle=2, n_boot=400)
    tr, te = probe.group_split(y, g)
    _, lo_i, hi_i = stats.bootstrap_ci(r.correct, n_boot=400, seed=0)
    _, lo_g, hi_g = stats.bootstrap_ci(r.correct, n_boot=400, seed=0, groups=g[te])
    assert r.acc_ci == (lo_g, hi_g)
    assert 0.0 <= lo_g <= r.acc <= hi_g <= 1.0 and (hi_i - lo_i) > 0


@quick
def test_data_and_pairs():
    from lablib import data

    ds = data.sentiment()
    assert len(ds["prompts"]) == len(ds["labels"]) == len(ds["groups"])
    assert set(ds["labels"]) == {0, 1}
    tf = data.true_false()
    assert sum(tf["labels"]) == len(tf["labels"]) // 2
    pairs = data.pairs_from_template("The {x} {animal} sat", ["black", "big"], ["white", "small"],
                                     animal=["cat", "dog"])
    assert len(pairs) == 4 and pairs[0] == ("The black cat sat", "The white cat sat")


# ---------------------------------------------------------------- cache key, saved argv
@quick
def test_acts_cache_key_covers_what_changes_the_numbers(monkeypatch):
    from types import SimpleNamespace

    from lablib import acts

    def model(commit="abc", dtype="float32", name="m"):
        return SimpleNamespace(name=name, dtype=dtype,
                               hf=SimpleNamespace(config=SimpleNamespace(_commit_hash=commit)))

    args = (["a prompt", "another"], [0, 5], "last")
    k = acts._key(model(), *args)
    assert k == acts._key(model(), *args) and len(k) == 24
    assert k == acts._key(model(), *args, max_length=512)  # the default is part of the key
    assert k != acts._key(model(), *args, max_length=64)
    assert k != acts._key(model(commit="def"), *args)
    assert k != acts._key(model(dtype="bfloat16"), *args)
    assert k != acts._key(model(name="other"), *args)
    assert k != acts._key(model(), ["a prompt", "another one"], [0, 5], "last")
    assert k != acts._key(model(), ["a prompt", "another"], [0, 6], "last")
    assert k != acts._key(model(), ["a prompt", "another"], [0, 5], "mean")
    monkeypatch.setattr(acts, "_versions", lambda: {"torch": "0.0.0", "transformers": "0.0.0"})
    assert k != acts._key(model(), *args)  # a library upgrade starts a fresh cache
    assert acts._key(SimpleNamespace(name="m", dtype="float32"), *args)  # no .hf: still a key


@quick
def test_acts_cache_is_used_and_lives_in_lab_cache(m, tmp_path, monkeypatch):
    from lablib import acts

    monkeypatch.setenv("LAB_CACHE", str(tmp_path))
    prompts = ["Hello there", "A second prompt"]
    a = acts.get_acts(m, prompts, [3], "last")
    files = list((tmp_path / "acts").glob("*.pt"))
    assert len(files) == 1
    b = acts.get_acts(m, prompts, [3], "last")  # read back from the file
    assert np.allclose(a[3], b[3])
    acts.get_acts(m, prompts, [3], "last", max_length=4)  # another truncation: another file
    assert len(list((tmp_path / "acts").glob("*.pt"))) == 2


@quick
def test_redact_argv():
    from lablib import report

    r = report.redact_argv
    assert r(["x.py", "--api-key", "sk-123", "--model", "gpt2"]) == ["x.py", "--api-key", "***", "--model", "gpt2"]
    assert r(["x.py", "--api-key=sk-123"]) == ["x.py", "--api-key=***"]
    assert r(["x.py", "--token", "abc", "--hf_token=abc", "HF_TOKEN=abc"]) == [
        "x.py", "--token", "***", "--hf_token=***", "HF_TOKEN=***"]
    assert r(["--password", "p", "--client-secret", "s", "--key", "k"]) == [
        "--password", "***", "--client-secret", "***", "--key", "***"]
    # counts of tokens and ordinary words are not secrets
    plain = ["x.py", "--max-new-tokens", "20", "--tokens=8", "--keyword", "cat", "--monkey", "3", "token"]
    assert r(plain) == plain
    assert r(["--api-key", "--model", "gpt2"]) == ["--api-key", "--model", "gpt2"]  # no value given: nothing to hide
    assert r([1, "--seed", 2]) == ["1", "--seed", "2"]


def test_saved_argv_is_redacted(tmp_path, monkeypatch):
    import json
    import sys as _sys

    from lablib import report

    monkeypatch.setenv("LAB_RUN_DIR", str(tmp_path / "r"))
    monkeypatch.setattr(_sys, "argv", ["nofile.py", "--api-key", "sk-not-a-real-key", "--seed", "1"])
    report.run_dir("x").save({"a": 1})
    text = (tmp_path / "r" / "results.json").read_text()
    assert "sk-not-a-real-key" not in text
    assert json.loads(text)["meta"]["argv"] == ["nofile.py", "--api-key", "***", "--seed", "1"]


# ---------------------------------------------------------------- model
@quick
def test_acts_shapes_and_padding(m):
    from lablib import acts

    prompts = ["Hello there", "The capital of France is Paris and"]
    a = acts.get_acts(m, prompts, [0, 5, -1], "last", cache=False)
    assert sorted(a) == [0, 5, 11]
    assert a[5].shape == (2, m.d_model)
    full = acts.get_acts(m, prompts, [5], "all", cache=False)[5]
    lens = m.encode(prompts)["lengths"]
    assert [t.shape[0] for t in full] == lens
    # batched (right-padded) == unbatched
    single = acts.get_acts(m, prompts[:1], [5], "last", cache=False)[5]
    assert np.allclose(single[0], a[5][0], atol=1e-4)
    # layer convention: layer L = HF hidden_states[L+1] (not the last one)
    import torch

    enc = m.encode(prompts[:1])
    with torch.no_grad():
        hs = m.hf(**{k: enc[k] for k in ("input_ids", "attention_mask")},
                  output_hidden_states=True).hidden_states
    assert torch.allclose(hs[6][0, -1], a[5][0], atol=1e-4)


@quick
def test_steering_changes_logits_and_random_same_norm(m):
    import torch

    from lablib import steer

    v = torch.randn(m.d_model, generator=torch.Generator().manual_seed(0))
    r = steer.random_like(v, 5)
    assert r.shape == (5, m.d_model)
    assert torch.allclose(r.norm(dim=1), v.norm().expand(5), rtol=1e-5)
    prompts = ["The weather today is", "I think the movie was"]
    base = steer.last_logits(m, prompts)
    with steer.steering(m, 6, v, alpha=10.0) as st:
        out = steer.last_logits(m, prompts, st)
    assert not torch.allclose(base, out, atol=1e-3)
    assert torch.allclose(base, steer.last_logits(m, prompts), atol=1e-5)  # hook removed
    met = steer.logit_diff_metric(m, " good", " bad")
    res = steer.compare_to_random(m, prompts, 6, v, 10.0, met, k=3)
    assert len(res["null"]) == 3 and np.isfinite(res["effect"])


@quick
def test_ablation_removes_projection(m):
    import torch

    from lablib import ablate, acts
    from lablib.models import block_hook

    u = torch.randn(m.d_model, generator=torch.Generator().manual_seed(0))
    u = u / u.norm()
    prompts = ["The weather today is", "A much longer prompt about cats and dogs"]
    seen = {}
    with ablate.ablation(m, [4], u, mode="project"):
        def rec(h):
            seen["h"] = h.detach().clone()
        # a second hook registered after the ablation sees the ablated output
        with block_hook(m, 4, rec):
            acts.run_body(m, m.encode(prompts))
    assert (seen["h"] @ u).abs().max() < 1e-3
    raw = acts.get_acts(m, prompts, [4], "all", cache=False)[4]
    assert (torch.cat(raw) @ u).abs().max() > 1e-2  # it was there before
    # mean mode holds the projection at the reference mean
    mu = ablate.mean_projection(m, prompts, 4, u)
    with ablate.ablation(m, [4], u, mode="mean", ref_prompts=prompts):
        with block_hook(m, 4, rec):
            acts.run_body(m, m.encode(prompts[:1]))
    assert torch.allclose(seen["h"] @ u, torch.full_like(seen["h"] @ u, mu), atol=1e-3)


def test_lens_shapes(m):
    from lablib import lens

    res = lens.compare(m, "The Eiffel Tower is in the city of", layers=[0, 6, 11], k=3,
                       target=" Paris")
    assert [r["layer"] for r in res["rows"]] == [0, 6, 11]
    for r in res["rows"]:
        assert len(r["logit_lens_top"]) == 3 and len(r["jlens_top"]) == 3
        assert 0 <= r["logit_rank"] < len(m.tok)
    assert res["rows"][-1]["logit_lens_top"] == res["rows"][-1]["jlens_top"]
    acts_, ids = lens.residuals(m, "Hello world", [0, 3])
    assert acts_[3].shape == (len(ids), m.d_model)


def test_patch_recovery_bounds(m):
    from lablib import patch, steer

    met = steer.logit_diff_metric(m, " Paris", " Rome")
    res = patch.patch_scan(m, "The capital of France is", "The capital of Italy is", met,
                           layers=[0, 11])
    rec = np.array(res["recovery"])
    assert rec.shape == (2, len(res["positions"]))
    assert rec[0, -2] == pytest.approx(1.0, abs=0.05)  # layer 0, country token: full recovery
    assert rec[-1, -1] == pytest.approx(1.0, abs=1e-4)  # last layer, last token: exact
    with pytest.raises(ValueError):
        patch.check_pair(m, "The capital of France is", "The capital of the United Kingdom is")


def test_report_run_dir(tmp_path, monkeypatch):
    import json

    from lablib import report

    monkeypatch.setenv("LAB_RUN_DIR", str(tmp_path / "r"))
    run = report.run_dir("x")
    assert run.path == tmp_path / "r"
    run.save({"a": np.float32(1.5), "nan": float("nan")}, rows=[{"k": 1}], model="gpt2", seed=3)
    j = json.loads((tmp_path / "r" / "results.json").read_text())
    assert j["results"]["a"] == 1.5 and j["results"]["nan"] is None
    assert j["meta"]["seed"] == 3 and j["meta"]["model"] == "gpt2" and "torch" in j["meta"]["versions"]
    assert "| k |" in (tmp_path / "r" / "results.md").read_text()
    monkeypatch.delenv("LAB_RUN_DIR")
    monkeypatch.chdir(tmp_path)
    p = report.run_dir("my run").path
    assert p.parent == tmp_path / "lab-runs" and p.name.endswith("-my-run")
