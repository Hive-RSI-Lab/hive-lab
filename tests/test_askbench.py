"""Tests for askbench.py. No real model calls: a fake `claude` stub is put first on PATH."""
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, "askbench.py")

spec_ = importlib.util.spec_from_file_location("askbench", SCRIPT)
ab = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(ab)

STUB = r'''#!/usr/bin/env python3
import json, os, sys
log = os.environ.get("STUB_LOG") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub_log.jsonl")
if "--version" in sys.argv:
    print("9.9.9 (stub)"); sys.exit(0)
prompt = sys.stdin.read()
args = sys.argv[1:]
system = args[args.index("--system-prompt") + 1]
entry = {"argv": args, "cwd": os.getcwd(), "cwd_listing": os.listdir("."),
         "env_keys": sorted(os.environ), "prompt": prompt, "system": system}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub_log.jsonl"), "a") as f:
    f.write(json.dumps(entry) + "\n")
if "FAIL_ME" in prompt:
    print("boom", file=sys.stderr); sys.exit(1)
if "<rubric>" in prompt:
    resp = prompt.split("<response>")[1]
    out = "Reasoning.\nANSWER: " + ("1" if "yes" in resp.lower() else "0")
elif "SAY_YES" in prompt:
    out = "yes"
elif "SAY_GARBLE" in prompt:
    out = "hmm"
else:
    out = "no"
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": out,
                  "session_id": "stub", "num_turns": 1, "duration_ms": 1,
                  "modelUsage": {"claude-stub": {}}, "usage": {"input_tokens": 1, "output_tokens": 1},
                  "credential_expires_at": "SHOULD-NOT-BE-SAVED"}))
'''


@pytest.fixture
def stub(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    p = bindir / "claude"
    p.write_text(STUB)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("AGENT_NAME", "secret-agent-name")  # must not reach the subject
    monkeypatch.delenv("ASKBENCH_CLAUDE", raising=False)
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    return bindir


def stub_log(bindir):
    p = bindir / "stub_log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def write_spec(tmp_path, body, name="exp.toml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


BASIC = """
name = "t"
model = "haiku"
samples = 3
seed = 1
system = "SYS {topic}"
[conditions.a]
prompt = "Q {topic} SAY_YES {{literal braces}}"
[conditions.b]
prompt = "Q {topic}"
[outcome]
type = "binary"
[score]
method = "keyword"
keywords = ["yes"]
[[items]]
id = "i1"
topic = "cats"
[[items]]
id = "i2"
topic = "dogs"
"""


# ------------------------------------------------------------------ statistics

def test_mean_and_percentile_known_values():
    assert ab.mean([1, 2, 3, 4]) == 2.5
    xs = [1, 2, 3, 4, 5]
    assert ab.percentile(xs, 0) == 1
    assert ab.percentile(xs, 50) == 3
    assert ab.percentile(xs, 100) == 5
    assert ab.percentile(xs, 25) == 2
    assert ab.percentile([0, 10], 2.5) == pytest.approx(0.25)  # numpy linear interpolation


def test_bootstrap_constant_and_coverage():
    assert ab.bootstrap_ci([[1, 1], [1, 1]]) == (1, 1)
    # binomial p=0.5, n=100 as 100 one-sample items: CI ~ 0.5 +- 1.96*0.05
    data = [[1]] * 50 + [[0]] * 50
    lo, hi = ab.bootstrap_ci(data, seed=3)
    assert 0.38 < lo < 0.42 and 0.58 < hi < 0.62


def test_bootstrap_clusters_by_item():
    # two items with opposite, perfectly consistent outcomes: resampling items
    # must give a much wider interval than treating 200 samples as independent
    lo, hi = ab.bootstrap_ci([[1] * 100, [0] * 100], seed=0)
    assert lo == 0.0 and hi == 1.0


def test_diff_ci_and_permutation_clear_effect():
    a = {str(i): [1] * 5 for i in range(6)}
    b = {str(i): [0] * 5 for i in range(6)}
    lo, hi = ab.bootstrap_diff_ci(a, b)
    assert lo == 1.0 and hi == 1.0
    p = ab.permutation_p(a, b, n_perm=2000)
    assert p == pytest.approx(1 / 2001)  # no shuffle reaches |diff| = 1


def test_permutation_no_effect():
    a = {"x": [1, 0, 1, 0]}
    b = {"x": [1, 0, 1, 0]}
    assert ab.permutation_p(a, b, n_perm=500) == 1.0


def test_permutation_exact_small_case():
    # A=[1,1], B=[0,0] in one stratum: of the 6 equally likely splits, 2 give |diff| = 1
    p = ab.permutation_p({"s": [1, 1]}, {"s": [0, 0]}, n_perm=20000, seed=5)
    assert p == pytest.approx(2 / 6, abs=0.015)


def test_cohen_kappa_known_values():
    # Wikipedia example: 50 items, agree yes 20, agree no 15, A yes/B no 5, A no/B yes 10 -> 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert ab.cohen_kappa(a, b) == pytest.approx(0.4)
    assert ab.cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert math.isnan(ab.cohen_kappa([1, 1], [1, 1]))


def test_tvd():
    f = ab.tvd_stat(["A", "B", "C"])
    assert f(["A", "A"], ["B", "B"]) == 1.0
    assert f(["A", "B"], ["B", "A"]) == 0.0
    assert f(["A", "A", "B", "B"], ["A", "A", "A", "C"]) == pytest.approx(0.5)


# ------------------------------------------------------------------ scoring

def _spec(outcome, score, **kw):
    s = {"model": "m", "conditions": {"c": {"prompt": "p"}}, "outcome": outcome, "score": score}
    s.update(kw)
    ab.validate_spec(s)
    return s


def test_score_choice_categorical():
    s = _spec({"type": "categorical", "categories": ["A", "B", "C"]}, {"method": "choice"})
    assert ab.score_rule(s, "B") == "B"
    assert ab.score_rule(s, "(C)") == "C"
    assert ab.score_rule(s, "**A**") == "A"
    assert ab.score_rule(s, "The answer is B, because") == "B"
    assert ab.score_rule(s, "Answer: C") == "C"
    assert ab.score_rule(s, "A) Venus is closest") == "A"
    assert ab.score_rule(s, "I am not sure") is None


def test_score_choice_binary_target():
    s = _spec({"type": "binary"}, {"method": "choice", "choices": ["A", "B"], "target_var": "wrong"})
    assert ab.score_rule(s, "B", {"wrong": "B"}) == 1
    assert ab.score_rule(s, "A", {"wrong": "B"}) == 0
    assert ab.score_rule(s, "dunno", {"wrong": "B"}) is None


def test_score_regex_and_keyword():
    s = _spec({"type": "scale", "min": 1, "max": 7}, {"method": "regex", "pattern": r"rating:\s*(\d+)"})
    assert ab.score_rule(s, "Rating: 5") == 5
    assert ab.score_rule(s, "Rating: 9") is None
    s = _spec({"type": "binary"}, {"method": "regex", "yes_pattern": r"\bcomply\b", "no_pattern": r"\brefuse\b"})
    assert ab.score_rule(s, "I comply") == 1
    assert ab.score_rule(s, "I refuse") == 0
    assert ab.score_rule(s, "comply or refuse") is None
    s = _spec({"type": "categorical", "categories": ["pos", "neg"]},
              {"method": "keyword", "keywords": {"pos": ["great"], "neg": ["awful"]}})
    assert ab.score_rule(s, "Great stuff") == "pos"
    assert ab.score_rule(s, "great and awful") is None


def test_parse_judge_answer():
    s = _spec({"type": "binary"}, {"method": "judge"}, judge={"rubric": "r"})
    assert ab.parse_judge_answer(s, "blah\nANSWER: 1") == 1
    assert ab.parse_judge_answer(s, "ANSWER: 0 ... ANSWER: yes") == 1
    assert ab.parse_judge_answer(s, "no answer line") is None


def test_template_fill_leaves_other_braces():
    assert ab.fill("a {x} {not set} {{y}}", {"x": 1}) == "a 1 {not set} {{y}}"
    with pytest.raises(ab.SpecError):
        ab.fill("{missing}", {})


def test_bad_specs_rejected(tmp_path):
    p = write_spec(tmp_path, BASIC.replace('type = "binary"', 'type = "nope"'))
    with pytest.raises(ab.SpecError):
        ab.load_spec(p)


# ------------------------------------------------------------------ end-to-end with the stub

def test_plan_is_seeded_and_complete(tmp_path):
    s = ab.load_spec(write_spec(tmp_path, BASIC))
    c1 = [c["id"] for c in ab.plan_calls(s)]
    c2 = [c["id"] for c in ab.plan_calls(s)]
    assert c1 == c2 and len(c1) == 12 and len(set(c1)) == 12
    assert c1 != sorted(c1)  # randomised


def test_dry_run_makes_no_calls(stub, tmp_path, capsys):
    spec = write_spec(tmp_path, BASIC)
    rd = tmp_path / "run"
    assert ab.main([spec, "--dry-run", "--run-dir", str(rd)]) == 0
    out = capsys.readouterr().out
    assert out.count("\n[") == 3 and ("SYS cats" in out or "SYS dogs" in out)
    assert "a: 6 subject calls" in out
    assert not stub_log(stub) and not rd.exists()


def test_cap_refuses_and_yes_calls_must_match(stub, tmp_path, capsys):
    spec = write_spec(tmp_path, BASIC)
    rd = str(tmp_path / "run")
    assert ab.main([spec, "--cap", "5", "--run-dir", rd]) == 3
    assert ab.main([spec, "--cap", "5", "--yes-calls", "11", "--run-dir", rd]) == 3
    assert not stub_log(stub)
    assert ab.main([spec, "--cap", "5", "--yes-calls", "12", "--run-dir", rd]) == 0
    assert len(stub_log(stub)) == 12


def test_full_run_records_flags_env_and_resume(stub, tmp_path, capsys):
    spec = write_spec(tmp_path, BASIC)
    rd = tmp_path / "run"
    assert ab.main([spec, "--run-dir", str(rd)]) == 0
    log = stub_log(stub)
    assert len(log) == 12
    for e in log:
        a = e["argv"]
        assert a[a.index("--tools") + 1] == ""
        assert a[a.index("--setting-sources") + 1] == ""
        for flag in ("--print", "--strict-mcp-config", "--safe-mode", "--no-session-persistence",
                     "--disable-slash-commands"):
            assert flag in a
        assert a[a.index("--output-format") + 1] == "json"
        assert a[a.index("--model") + 1] == "haiku"
        assert e["cwd_listing"] == []
        assert e["cwd"].startswith(str(tmp_path / "scratch"))
        assert "AGENT_NAME" not in e["env_keys"] and "CLAUDECODE" not in e["env_keys"]
    recs = [json.loads(l) for l in (rd / "calls.jsonl").read_text().splitlines()]
    assert len(recs) == 12
    r = recs[0]
    for k in ("prompt", "system", "model", "output", "outcome", "duration_s", "exit_status", "judges"):
        assert k in r
    assert "SHOULD-NOT-BE-SAVED" not in (rd / "calls.jsonl").read_text()
    a_out = {x["outcome"] for x in recs if x["condition"] == "a"}
    b_out = {x["outcome"] for x in recs if x["condition"] == "b"}
    assert a_out == {1} and b_out == {0}
    assert "{literal braces}" in [x for x in recs if x["condition"] == "a"][0]["prompt"]
    rep = (rd / "report.md").read_text()
    assert "| a | 6 | 0 | 0 (0.0%) | 6 | 100.0% |" in rep
    assert "| a | b | +100.0% |" in rep
    assert (rd / "spec.toml").exists() and json.loads((rd / "plan.json").read_text())["seed"] == 1
    # resume: nothing more to do
    assert ab.main([spec, "--run-dir", str(rd)]) == 0
    assert len(stub_log(stub)) == 12


def test_failed_calls_are_retried_on_rerun(stub, tmp_path):
    body = BASIC.replace('prompt = "Q {topic}"', 'prompt = "Q {topic} {flag}"')
    body = body.replace('topic = "dogs"', 'topic = "dogs"\nflag = "FAIL_ME"').replace('topic = "cats"', 'topic = "cats"\nflag = ""')
    spec = write_spec(tmp_path, body)
    rd = tmp_path / "run"
    assert ab.main([spec, "--run-dir", str(rd)]) == 0
    assert len(stub_log(stub)) == 12
    rep = (rd / "report.md").read_text()
    assert "| b | 6 | 3 |" in rep and "calls failed" in rep
    ab.main([spec, "--run-dir", str(rd)])
    assert len(stub_log(stub)) == 15  # only the 3 failed calls ran again


def test_judge_is_blind_and_double(stub, tmp_path):
    body = """
    name = "j"
    model = "haiku"
    samples = 2
    seed = 2
    [conditions.secret_cond_alpha]
    prompt = "ALPHA-PROMPT SAY_YES {x}"
    [conditions.secret_cond_beta]
    prompt = "BETA-PROMPT {x}"
    [outcome]
    type = "binary"
    [score]
    method = "judge"
    [judge]
    rubric = "1 if the reply says yes"
    repeats = 2
    include_vars = ["x"]
    [[items]]
    x = "item-var-value"
    """
    spec = write_spec(tmp_path, body)
    rd = tmp_path / "run"
    assert ab.main([spec, "--run-dir", str(rd)]) == 0
    log = stub_log(stub)
    judge_calls = [e for e in log if "<rubric>" in e["prompt"]]
    assert len(judge_calls) == 8
    for e in judge_calls:
        assert "secret_cond" not in e["prompt"] and "ALPHA-PROMPT" not in e["prompt"] \
            and "BETA-PROMPT" not in e["prompt"]
        assert "item-var-value" in e["prompt"]
    rep = (rd / "report.md").read_text()
    assert "judge run 1 vs judge run 2: 4 pairs, exact agreement 100.0%" in rep


def test_unparseable_share_reported(stub, tmp_path):
    body = BASIC.replace('prompt = "Q {topic}"', 'prompt = "Q {topic} SAY_GARBLE"').replace(
        'keywords = ["yes"]', 'keywords = ["yes"]\nno_keywords = ["no"]')
    spec = write_spec(tmp_path, body)
    rd = tmp_path / "run"
    assert ab.main([spec, "--run-dir", str(rd)]) == 0
    rep = (rd / "report.md").read_text()
    assert "| b | 6 | 0 | 6 (100.0%) | 0 |" in rep
    assert "unparseable" in rep


def test_cli_entrypoint_runs(stub, tmp_path):
    spec = write_spec(tmp_path, BASIC)
    p = subprocess.run([sys.executable, SCRIPT, spec, "--dry-run", "--run-dir", str(tmp_path / "r")],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "total calls this invocation: 12" in p.stdout


def test_examples_validate():
    ex = os.path.join(ROOT, "askbench-examples")
    for f in os.listdir(ex):
        if f.endswith((".toml", ".json")):
            ab.load_spec(os.path.join(ex, f))
