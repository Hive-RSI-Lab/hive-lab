#!/usr/bin/env python3
"""askbench: small behavioural experiments on a Claude model, through the `claude` CLI.

Run every (condition, item, sample) as an independent, clean `claude --print` call,
score each output (rule first, optional blind LLM judge), and report rates/means with
bootstrap 95% intervals, pairwise differences and permutation p-values.

Stdlib only. See ASKBENCH.md next to this file.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from datetime import datetime, timezone

VERSION = "0.1"
DEFAULT_CAP = 150
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 180
N_BOOT = 2000
N_PERM = 5000

# Flags that make the subject as clean as the CLI allows (verified 2026-11-14, CLI 2.1.325):
#   --tools ""               no built-in tools
#   --strict-mcp-config      no MCP servers (none given via --mcp-config)
#   --setting-sources ""     no user/project/local settings (so no hooks from them)
#   --safe-mode              no CLAUDE.md, skills, plugins, hooks, MCP, custom agents
#   --disable-slash-commands no skills/commands
#   --no-session-persistence session not saved to disk
#   --system-prompt          replaces the default Claude Code system prompt
# --bare would be stricter but only accepts ANTHROPIC_API_KEY auth, not the signed-in CLI.
CLEAN_FLAGS = [
    "--print",
    "--tools", "",
    "--strict-mcp-config",
    "--setting-sources", "",
    "--safe-mode",
    "--disable-slash-commands",
    "--no-session-persistence",
    "--output-format", "json",
]

# Only these environment variables reach the subject process.
ENV_ALLOW = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
             "TMPDIR", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
             "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "SHELL", "TZ")

# Fields kept from the CLI's JSON result (others, e.g. credential metadata, are dropped).
JSON_KEEP = ("type", "subtype", "is_error", "result", "stop_reason", "terminal_reason",
             "num_turns", "duration_ms", "duration_api_ms", "session_id",
             "permission_denials", "api_error_status")

JUDGE_SYSTEM = (
    "You are a careful grader. You will be given a rubric and a response written by an AI "
    "model. Apply the rubric to the response only. Do not answer the task yourself. "
    "Think briefly if needed, then end with a final line of the form `ANSWER: <value>`."
)


class SpecError(Exception):
    pass


_SECRET_NAME = re.compile(r"(^|[-_])(api)?key$|(^|[-_])token$|token[-_]?(id|secret)$|secret|passw|credential",
                          re.I)


def redact_argv(argv) -> list:
    """argv with the value of any option whose name looks like a key, token, secret or password replaced by ***."""
    out, hide = [], False
    for a in argv:
        a = str(a)
        if hide and not a.startswith("--"):
            out.append("***")
            hide = False
            continue
        hide = False
        name, eq, _ = a.partition("=")
        if _SECRET_NAME.search(name.lstrip("-")):
            if eq:
                a = name + "=***"
            elif a.startswith("-"):
                hide = True
        out.append(a)
    return out


# --------------------------------------------------------------------------- spec

def load_spec(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    if path.endswith(".json"):
        spec = json.loads(raw.decode())
    else:
        spec = tomllib.loads(raw.decode())
    spec["_sha"] = hashlib.sha256(raw).hexdigest()[:10]
    spec["_path"] = os.path.abspath(path)
    spec["_raw"] = raw.decode()
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    for k in ("model", "conditions", "outcome", "score"):
        if k not in spec:
            raise SpecError(f"spec is missing `{k}`")
    spec.setdefault("name", os.path.splitext(os.path.basename(spec.get("_path", "spec")))[0])
    spec.setdefault("samples", 10)
    spec.setdefault("seed", 12345)
    if not isinstance(spec["samples"], int) or spec["samples"] < 1:
        raise SpecError("`samples` must be a positive integer")
    conds = spec["conditions"]
    if not isinstance(conds, dict) or not conds:
        raise SpecError("`conditions` must be a table of named conditions")
    for name, c in conds.items():
        if "|" in name:
            raise SpecError(f"condition name {name!r} may not contain '|'")
        if not isinstance(c, dict):
            raise SpecError(f"condition {name!r} must be a table")
        if "prompt" not in c and "prompt" not in spec:
            raise SpecError(f"condition {name!r} has no `prompt` and there is no top-level `prompt`")
    items = spec.get("items") or [{}]
    ids = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise SpecError("each item must be a table of variables")
        ids.append(str(it.get("id", i)))
    if len(set(ids)) != len(ids):
        raise SpecError("item ids must be unique")
    spec["items"] = items
    out = spec["outcome"]
    t = out.get("type")
    if t not in ("binary", "categorical", "scale"):
        raise SpecError("outcome.type must be binary, categorical or scale")
    if t == "categorical" and not out.get("categories"):
        raise SpecError("categorical outcome needs outcome.categories")
    if t == "scale" and ("min" not in out or "max" not in out):
        raise SpecError("scale outcome needs outcome.min and outcome.max")
    sc = spec["score"]
    m = sc.get("method")
    if m not in ("regex", "keyword", "choice", "judge"):
        raise SpecError("score.method must be regex, keyword, choice or judge")
    if m == "choice":
        if t == "categorical":
            sc.setdefault("choices", out["categories"])
        elif t == "binary":
            if not sc.get("choices") or not sc.get("target_var"):
                raise SpecError("choice scoring with a binary outcome needs score.choices and score.target_var "
                                "(outcome = 1 when the chosen letter equals the item's target_var)")
        else:
            raise SpecError("choice scoring needs a categorical or binary outcome")
    if m == "regex" and not (sc.get("pattern") or sc.get("yes_pattern")):
        raise SpecError("regex scoring needs score.pattern (or yes_pattern/no_pattern for binary)")
    if m == "keyword" and not sc.get("keywords"):
        raise SpecError("keyword scoring needs score.keywords")
    j = spec.get("judge")
    if m == "judge" and not j:
        raise SpecError("score.method = judge needs a [judge] table with a rubric")
    if j is not None:
        if not j.get("rubric"):
            raise SpecError("judge.rubric is required")
        j.setdefault("model", spec["model"])
        j.setdefault("use", "always" if m == "judge" else "fallback")
        if j["use"] not in ("fallback", "always"):
            raise SpecError("judge.use must be fallback or always")
        j.setdefault("repeats", 1)
        if j["repeats"] not in (1, 2):
            raise SpecError("judge.repeats must be 1 or 2")
        j.setdefault("include_vars", [])
    # check templates render for every item
    for name in conds:
        for it in items:
            render_call(spec, name, it)


_VAR = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def fill(template: str, variables: dict) -> str:
    """Replace {name} with variables[name]. Other braces, and {{name}}, are left as written."""
    def rep(m):
        k = m.group(1)
        if k not in variables:
            raise SpecError(f"template uses {{{k}}} but the item has no `{k}`")
        return str(variables[k])
    return _VAR.sub(rep, template)


def render_call(spec: dict, cond: str, item: dict) -> tuple[str, str]:
    c = spec["conditions"][cond]
    system = c.get("system", spec.get("system", ""))
    prompt = c.get("prompt", spec.get("prompt"))
    return fill(system, item), fill(prompt, item)


def item_id(item: dict, idx: int) -> str:
    return str(item.get("id", idx))


def plan_calls(spec: dict) -> list[dict]:
    calls = []
    for cond in spec["conditions"]:
        for idx, it in enumerate(spec["items"]):
            system, prompt = render_call(spec, cond, it)
            for s in range(spec["samples"]):
                calls.append({
                    "id": f"{cond}|{item_id(it, idx)}|{s}",
                    "condition": cond, "item": item_id(it, idx), "sample": s,
                    "vars": it, "system": system, "prompt": prompt,
                })
    random.Random(spec["seed"]).shuffle(calls)
    return calls


# --------------------------------------------------------------------------- scoring

def _norm_cat(value: str, categories: list) -> str | None:
    v = value.strip().strip(".()[]*:\"'").lower()
    for c in categories:
        if str(c).lower() == v:
            return str(c)
    return None


def _num_in_range(text: str, lo, hi):
    try:
        x = float(text)
    except (TypeError, ValueError):
        return None
    if lo <= x <= hi:
        return int(x) if x == int(x) else x
    return None


def score_rule(spec: dict, text: str, item: dict | None = None):
    """Return the parsed outcome, or None if the rule cannot parse it."""
    sc, out = spec["score"], spec["outcome"]
    t, m = out["type"], sc["method"]
    item = item or {}
    if text is None:
        return None
    flags = 0 if sc.get("case_sensitive") else re.IGNORECASE
    if m == "judge":
        return None
    if m == "regex":
        if t == "binary":
            if sc.get("yes_pattern"):
                y = re.search(sc["yes_pattern"], text, flags)
                n = re.search(sc["no_pattern"], text, flags) if sc.get("no_pattern") else None
                if sc.get("no_pattern"):
                    if bool(y) == bool(n):
                        return None
                    return 1 if y else 0
                return 1 if y else 0
            return 1 if re.search(sc["pattern"], text, flags) else 0
        mt = re.search(sc["pattern"], text, flags)
        if not mt:
            return None
        g = mt.group(1) if mt.groups() else mt.group(0)
        if t == "categorical":
            return _norm_cat(g, out["categories"])
        return _num_in_range(g, out["min"], out["max"])
    if m == "keyword":
        kw = sc["keywords"]
        low = text if sc.get("case_sensitive") else text.lower()

        def has(words):
            return any((w if sc.get("case_sensitive") else w.lower()) in low for w in words)
        if t == "binary":
            yes = has(kw if isinstance(kw, list) else kw.get("yes", []))
            no_words = sc.get("no_keywords") or ([] if isinstance(kw, list) else kw.get("no", []))
            if no_words:
                no = has(no_words)
                if yes == no:
                    return None
                return 1 if yes else 0
            return 1 if yes else 0
        if t == "categorical":
            hits = [c for c, words in kw.items() if has(words)]
            return _norm_cat(hits[0], out["categories"]) if len(hits) == 1 else None
        raise SpecError("keyword scoring is not defined for scale outcomes; use regex")
    if m == "choice":
        letters = [str(c) for c in sc["choices"]]
        alt = "|".join(re.escape(x) for x in letters)
        s = text.strip()
        pats = [
            rf"^\**\(?({alt})\)?\**[.):]?\s*$",
            rf"(?i:answer)\s*(?:is|:)?\s*[:\-]?\s*\**\(?({alt})\)?\**(?![A-Za-z])",
            rf"^\**\(?({alt})\)?\**[.):]\s",
        ]
        for p in pats:
            mt = re.search(p, s, re.MULTILINE)
            if mt:
                letter = mt.group(1)
                if t == "binary":
                    return 1 if letter == str(item.get(sc["target_var"])) else 0
                return letter
        return None
    return None


def parse_judge_answer(spec: dict, text: str):
    if not text:
        return None
    out = spec["outcome"]
    parts = re.split(r"ANSWER:", text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    lines = parts[-1].strip().splitlines()
    v = lines[0].strip().strip("`*").strip() if lines else ""
    t = out["type"]
    if t == "binary":
        lv = v.lower().strip(".")
        if lv in ("1", "yes", "true"):
            return 1
        if lv in ("0", "no", "false"):
            return 0
        return None
    if t == "categorical":
        return _norm_cat(v, out["categories"])
    mt = re.match(r"-?\d+(?:\.\d+)?", v)
    return _num_in_range(mt.group(0), out["min"], out["max"]) if mt else None


def judge_prompt(spec: dict, call: dict, response: str) -> str:
    j, out = spec["judge"], spec["outcome"]
    if out["type"] == "binary":
        fmt = "ANSWER: 1 (criterion met) or ANSWER: 0 (not met)"
    elif out["type"] == "categorical":
        fmt = "ANSWER: one of " + ", ".join(str(c) for c in out["categories"])
    else:
        fmt = f"ANSWER: a number from {out['min']} to {out['max']}"
    parts = ["<rubric>\n" + j["rubric"].strip() + "\n</rubric>"]
    for k in j.get("include_vars", []):
        if k in call["vars"]:
            parts.append(f"<{k}>\n{call['vars'][k]}\n</{k}>")
    parts.append("<response>\n" + (response or "") + "\n</response>")
    parts.append(f"End with exactly one final line: {fmt}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- calling

def clean_env(workdir: str | None = None) -> dict:
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    if workdir:  # stop git discovery above the empty working directory
        env["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.abspath(workdir))
    return env


def claude_cmd(model: str, system: str, effort: str | None) -> list[str]:
    exe = os.environ.get("ASKBENCH_CLAUDE", "claude")
    cmd = [exe] + CLEAN_FLAGS + ["--model", model, "--system-prompt", system]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def call_claude(model: str, system: str, prompt: str, effort: str | None,
                workdir: str, timeout: int) -> dict:
    """One clean call. The prompt goes in on stdin. Returns a record fragment."""
    if os.listdir(workdir):
        raise RuntimeError(f"working directory {workdir} is not empty")
    cmd = claude_cmd(model, system, effort)
    t0 = time.monotonic()
    rec = {"argv": cmd[1:]}
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           cwd=workdir, env=clean_env(workdir), timeout=timeout)
        rec["exit_status"] = p.returncode
        stdout, stderr = p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rec["exit_status"] = "timeout"
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    rec["duration_s"] = round(time.monotonic() - t0, 3)
    rec["stderr_tail"] = (stderr or "")[-2000:]
    parsed = None
    try:
        parsed = json.loads(stdout) if stdout.strip() else None
    except json.JSONDecodeError:
        pass
    if isinstance(parsed, dict):
        rec["cli"] = {k: parsed[k] for k in JSON_KEEP if k in parsed}
        rec["model_used"] = list((parsed.get("modelUsage") or {}).keys())
        u = parsed.get("usage") or {}
        rec["tokens"] = {k: u.get(k) for k in ("input_tokens", "output_tokens",
                                                 "cache_creation_input_tokens",
                                                 "cache_read_input_tokens")}
        rec["output"] = parsed.get("result")
        rec["ok"] = rec["exit_status"] == 0 and not parsed.get("is_error") and rec["output"] is not None
    else:
        rec["raw_stdout"] = (stdout or "")[-4000:]
        rec["output"] = None
        rec["ok"] = False
    for leftover in os.listdir(workdir):  # the subject has no tools, but be sure
        path = os.path.join(workdir, leftover)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    return rec


def run_one(spec: dict, call: dict, effort, timeout: int, workroot: str) -> dict:
    wd = tempfile.mkdtemp(prefix="subj-", dir=workroot)
    try:
        sub = call_claude(spec["model"], call["system"], call["prompt"], effort, wd, timeout)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    rec = {
        "id": call["id"], "condition": call["condition"], "item": call["item"],
        "sample": call["sample"], "vars": call["vars"], "model": spec["model"],
        "system": call["system"], "prompt": call["prompt"],
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rec.update(sub)
    rule = score_rule(spec, rec["output"], call["vars"]) if rec["ok"] else None
    rec["rule_outcome"] = rule
    j = spec.get("judge")
    judges = []
    if rec["ok"] and j and (j["use"] == "always" or rule is None):
        jp = judge_prompt(spec, call, rec["output"])
        for _ in range(j["repeats"]):
            wd = tempfile.mkdtemp(prefix="judge-", dir=workroot)
            try:
                jr = call_claude(j["model"], JUDGE_SYSTEM, jp, j.get("effort"), wd, timeout)
            finally:
                shutil.rmtree(wd, ignore_errors=True)
            jr["model"] = j["model"]
            jr["system"] = JUDGE_SYSTEM
            jr["prompt"] = jp
            jr["outcome"] = parse_judge_answer(spec, jr.get("output")) if jr["ok"] else None
            judges.append(jr)
    rec["judges"] = judges
    if j and j["use"] == "always":
        rec["outcome"] = judges[0]["outcome"] if judges else None
        rec["outcome_source"] = "judge"
    elif rule is not None:
        rec["outcome"], rec["outcome_source"] = rule, "rule"
    elif judges:
        rec["outcome"], rec["outcome_source"] = judges[0]["outcome"], "judge"
    else:
        rec["outcome"], rec["outcome_source"] = None, None
    return rec


# --------------------------------------------------------------------------- statistics

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(sorted_xs, q):
    """Linear-interpolation percentile (numpy default) of a sorted list, q in [0,100]."""
    if not sorted_xs:
        return float("nan")
    pos = (len(sorted_xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def bootstrap_ci(groups, stat=mean, n_boot=N_BOOT, seed=0, alpha=0.05):
    """Cluster bootstrap: `groups` is a list of lists (one list per item).
    Resamples items with replacement, pools their values, applies `stat`."""
    groups = [g for g in groups if g]
    if not groups:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(groups)
    vals = []
    for _ in range(n_boot):
        pool = []
        for _ in range(k):
            pool.extend(groups[rng.randrange(k)])
        vals.append(stat(pool))
    vals.sort()
    return (percentile(vals, 100 * alpha / 2), percentile(vals, 100 * (1 - alpha / 2)))


def bootstrap_diff_ci(a_groups: dict, b_groups: dict, stat=mean, n_boot=N_BOOT, seed=0, alpha=0.05):
    """CI for stat(A) - stat(B). Resamples shared item ids jointly (paired design).
    Groups are dicts item_id -> list of values."""
    keys = sorted(set(a_groups) | set(b_groups))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pa, pb = [], []
        for _ in range(len(keys)):
            k = keys[rng.randrange(len(keys))]
            pa.extend(a_groups.get(k, []))
            pb.extend(b_groups.get(k, []))
        if pa and pb:
            vals.append(stat(pa) - stat(pb))
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return (percentile(vals, 100 * alpha / 2), percentile(vals, 100 * (1 - alpha / 2)))


def permutation_p(a_groups: dict, b_groups: dict, stat=mean, n_perm=N_PERM, seed=0):
    """Two-sided permutation p-value for |stat(A) - stat(B)|, shuffling the condition
    label within each item (stratified). p = (hits + 1) / (n_perm + 1)."""
    keys = sorted(set(a_groups) | set(b_groups))
    a_all = [x for k in keys for x in a_groups.get(k, [])]
    b_all = [x for k in keys for x in b_groups.get(k, [])]
    if not a_all or not b_all:
        return float("nan")
    obs = abs(stat(a_all) - stat(b_all))
    strata = [(a_groups.get(k, []) + b_groups.get(k, []), len(a_groups.get(k, []))) for k in keys]
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        pa, pb = [], []
        for pool, na in strata:
            pool = pool[:]
            rng.shuffle(pool)
            pa.extend(pool[:na])
            pb.extend(pool[na:])
        if abs(stat(pa) - stat(pb)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b), key=str)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1:  # both raters used one and the same category: kappa is undefined
        return float("nan")
    return (po - pe) / (1 - pe)


def share_of(cat):
    return lambda xs: mean([1 if x == cat else 0 for x in xs])


def tvd_stat(categories):
    """Total variation distance between two category distributions (0..1)."""
    def f(pa, pb):
        return 0.5 * sum(abs(share_of(c)(pa) - share_of(c)(pb)) for c in categories)
    return f


# --------------------------------------------------------------------------- report

def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.3g}"


def by_condition_item(records, conditions):
    out = {c: {} for c in conditions}
    for r in records:
        if r.get("outcome") is not None and r["condition"] in out:
            out[r["condition"]].setdefault(r["item"], []).append(r["outcome"])
    return out


def build_report(spec: dict, records: list, run_dir: str) -> str:
    conds = list(spec["conditions"])
    t = spec["outcome"]["type"]
    cats = [str(c) for c in spec["outcome"].get("categories", [])]
    pct = t == "binary"
    seed = spec["seed"]
    groups = by_condition_item(records, conds)
    n_items = len(spec["items"])
    L = []
    W = []
    L.append(f"# askbench report: {spec['name']}")
    L.append("")
    L.append(f"- run folder: `{run_dir}`")
    L.append(f"- spec sha256[:10]: `{spec['_sha']}`; seed {seed}; subject model `{spec['model']}`")
    used = sorted({m for r in records for m in r.get("model_used", [])})
    if used:
        L.append(f"- model ids reported by the CLI: {', '.join('`'+u+'`' for u in used)}")
    L.append(f"- outcome: {t}; scoring: {spec['score']['method']}"
             + (f" + judge `{spec['judge']['model']}` ({spec['judge']['use']}, x{spec['judge']['repeats']})"
                if spec.get("judge") else ""))
    L.append(f"- items: {n_items}; samples per item per condition: {spec['samples']}")
    L.append("- intervals: 95% percentile bootstrap"
             + (" resampling items (clustered)" if n_items >= 5 else " resampling items; with <5 items they understate uncertainty")
             + f", {N_BOOT} resamples; p-values: permutation within item, {N_PERM} shuffles, two-sided")
    L.append("")
    L.append("## Per condition")
    L.append("")
    if t == "categorical":
        L.append("| condition | calls | failed | unparseable | n scored | " + " | ".join(f"{c} (95% CI)" for c in cats) + " |")
        L.append("|" + "---|" * (5 + len(cats)))
    else:
        L.append(f"| condition | calls | failed | unparseable | n scored | {'rate' if pct else 'mean'} | 95% CI |")
        L.append("|---|---|---|---|---|---|---|")
    for c in conds:
        rs = [r for r in records if r["condition"] == c]
        failed = sum(1 for r in rs if not r.get("ok"))
        unp = sum(1 for r in rs if r.get("ok") and r.get("outcome") is None)
        ok = len(rs) - failed
        vals = [x for g in groups[c].values() for x in g]
        n = len(vals)
        unp_share = unp / ok if ok else float("nan")
        if t == "categorical":
            cells = []
            for k in cats:
                st = share_of(k)
                lo, hi = bootstrap_ci(list(groups[c].values()), st, seed=seed)
                cells.append(f"{fmt(st(vals), True)} ({fmt(lo, True)}–{fmt(hi, True)})" if vals else "n/a")
            L.append(f"| {c} | {len(rs)} | {failed} | {unp} ({fmt(unp_share, True)}) | {n} | " + " | ".join(cells) + " |")
        else:
            lo, hi = bootstrap_ci(list(groups[c].values()), seed=seed)
            L.append(f"| {c} | {len(rs)} | {failed} | {unp} ({fmt(unp_share, True)}) | {n} | {fmt(mean(vals), pct)} | {fmt(lo, pct)} – {fmt(hi, pct)} |")
        if n < 30:
            W.append(f"`{c}` has only {n} scored outputs; intervals are wide and the bootstrap is rough below ~30.")
        if t == "binary" and vals and mean(vals) in (0.0, 1.0):
            W.append(f"`{c}` is at {fmt(mean(vals), True)}; a bootstrap interval of zero width does not mean certainty "
                     f"(rule of three: the true rate could be up to ~{fmt(min(1.0, 3 / n), True)} away).")
        if ok and unp_share > 0.1:
            W.append(f"`{c}`: {fmt(unp_share, True)} of outputs were unparseable and are excluded; check the scorer.")
        if failed:
            W.append(f"`{c}`: {failed} calls failed (see `exit_status`/`stderr_tail` in calls.jsonl); rerun to retry them.")
    L.append("")
    if len(conds) >= 2:
        L.append("## Pairwise differences (A minus B)")
        L.append("")
        if t == "categorical":
            L.append("| A | B | total variation distance | 95% CI | permutation p | largest share shift |")
            L.append("|---|---|---|---|---|---|")
        else:
            L.append(f"| A | B | difference in {'rate' if pct else 'mean'} | 95% CI | permutation p |")
            L.append("|---|---|---|---|---|")
        n_pairs = 0
        for i in range(len(conds)):
            for j in range(i + 1, len(conds)):
                a, b = conds[i], conds[j]
                ga, gb = groups[a], groups[b]
                if not ga or not gb:
                    continue
                n_pairs += 1
                if t == "categorical":
                    tv = tvd_stat(cats)
                    pa = [x for g in ga.values() for x in g]
                    pb = [x for g in gb.values() for x in g]
                    obs = tv(pa, pb)
                    lo, hi = _paired_boot_pair(ga, gb, tv, seed)
                    p = _perm_pair(ga, gb, tv, seed)
                    shifts = sorted(((share_of(k)(pa) - share_of(k)(pb), k) for k in cats), key=lambda z: -abs(z[0]))
                    d, k = shifts[0]
                    L.append(f"| {a} | {b} | {fmt(obs, True)} | {fmt(lo, True)} – {fmt(hi, True)} | {p:.4f} | {k}: {'+' if d >= 0 else ''}{fmt(d, True)} |")
                else:
                    pa = [x for g in ga.values() for x in g]
                    pb = [x for g in gb.values() for x in g]
                    d = mean(pa) - mean(pb)
                    lo, hi = bootstrap_diff_ci(ga, gb, seed=seed)
                    p = permutation_p(ga, gb, seed=seed)
                    L.append(f"| {a} | {b} | {'+' if d >= 0 else ''}{fmt(d, pct)} | {fmt(lo, pct)} – {fmt(hi, pct)} | {p:.4f} |")
        if n_pairs > 1:
            W.append(f"{n_pairs} pairwise comparisons: p-values are not corrected for multiple comparisons "
                     f"(Bonferroni threshold for 0.05 would be {0.05 / n_pairs:.4f}).")
        if t == "categorical":
            W.append("For categorical outcomes the total variation distance is never negative, so its bootstrap "
                     "interval is biased upward; rely on the permutation p-value for 'is there any difference'.")
        L.append("")
    # judge agreement
    j = spec.get("judge")
    if j:
        L.append("## Judge")
        L.append("")
        judged = [r for r in records if r.get("judges")]
        L.append(f"- outputs sent to the judge: {len(judged)}; judge calls: {sum(len(r['judges']) for r in judged)}; "
                 f"judge unparseable: {sum(1 for r in judged for x in r['judges'] if x.get('outcome') is None)}")
        L.append("- the judge saw only the rubric, the subject's output"
                 + (f" and item variables {j['include_vars']}" if j.get("include_vars") else "")
                 + "; never the condition name or the condition's prompt")
        if j["repeats"] == 2:
            pairs = [(r["judges"][0]["outcome"], r["judges"][1]["outcome"]) for r in judged
                     if len(r["judges"]) == 2 and None not in (r["judges"][0]["outcome"], r["judges"][1]["outcome"])]
            _agreement_lines(L, "judge run 1 vs judge run 2", pairs, t)
        rv = [(r["rule_outcome"], r["judges"][0]["outcome"]) for r in judged
              if r.get("rule_outcome") is not None and r["judges"][0].get("outcome") is not None]
        if rv:
            _agreement_lines(L, "rule vs judge", rv, t)
        L.append("")
    if W:
        L.append("## Warnings")
        L.append("")
        for w in W:
            L.append(f"- {w}")
        L.append("")
    L.append("## Files")
    L.append("")
    L.append("- `calls.jsonl`: one line per subject call (prompt, system prompt, raw output, parsed outcome, judge exchange, duration, exit status)")
    L.append("- `spec.toml` / `spec.json`: the exact spec; `plan.json`: seed, flags, CLI version, call order")
    return "\n".join(L) + "\n"


def _agreement_lines(L, label, pairs, t):
    if not pairs:
        L.append(f"- {label}: no pairs with both values")
        return
    agree = sum(1 for x, y in pairs if x == y) / len(pairs)
    line = f"- {label}: {len(pairs)} pairs, exact agreement {fmt(agree, True)}"
    if t in ("binary", "categorical"):
        line += f", Cohen's kappa {fmt(cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]))}"
    else:
        line += f", mean absolute difference {fmt(mean([abs(x - y) for x, y in pairs]))}"
    L.append(line)


def _paired_boot_pair(ga, gb, f, seed, n_boot=N_BOOT):
    keys = sorted(set(ga) | set(gb))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pa, pb = [], []
        for _ in range(len(keys)):
            k = keys[rng.randrange(len(keys))]
            pa.extend(ga.get(k, []))
            pb.extend(gb.get(k, []))
        if pa and pb:
            vals.append(f(pa, pb))
    vals.sort()
    return (percentile(vals, 2.5), percentile(vals, 97.5)) if vals else (float("nan"),) * 2


def _perm_pair(ga, gb, f, seed, n_perm=N_PERM):
    keys = sorted(set(ga) | set(gb))
    pa0 = [x for k in keys for x in ga.get(k, [])]
    pb0 = [x for k in keys for x in gb.get(k, [])]
    obs = f(pa0, pb0)
    strata = [(ga.get(k, []) + gb.get(k, []), len(ga.get(k, []))) for k in keys]
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        pa, pb = [], []
        for pool, na in strata:
            pool = pool[:]
            rng.shuffle(pool)
            pa.extend(pool[:na])
            pb.extend(pool[na:])
        if f(pa, pb) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- run folder

def read_records(path: str) -> list:
    recs = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line from an interrupted run
                recs[r["id"]] = r  # later lines win (retries)
    return list(recs.values())


def cli_version() -> str:
    try:
        p = subprocess.run([os.environ.get("ASKBENCH_CLAUDE", "claude"), "--version"],
                           capture_output=True, text=True, timeout=30, env=clean_env())
        return p.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e.__class__.__name__})"


def max_judge_calls(spec: dict, n_subject: int) -> int:
    j = spec.get("judge")
    return n_subject * j["repeats"] if j else 0


def print_plan(spec, todo, done_n, out=None):
    out = out or sys.stdout
    by = {}
    for c in todo:
        by[c["condition"]] = by.get(c["condition"], 0) + 1
    print(f"plan for '{spec['name']}': {len(spec['conditions'])} conditions x {len(spec['items'])} items "
          f"x {spec['samples']} samples; model {spec['model']}; seed {spec['seed']}", file=out)
    if done_n:
        print(f"  already finished in this run folder: {done_n} calls (skipped)", file=out)
    for c in spec["conditions"]:
        print(f"  {c}: {by.get(c, 0)} subject calls", file=out)
    mj = max_judge_calls(spec, len(todo))
    if mj:
        print(f"  judge calls: up to {mj} ({spec['judge']['use']}, x{spec['judge']['repeats']}, model {spec['judge']['model']})", file=out)
    total = len(todo) + mj
    print(f"  total calls this invocation: {total}" + (" (upper bound)" if mj else ""), file=out)
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="askbench", description=__doc__.split("\n\n")[0],
                                 epilog="Full guide: ASKBENCH.md next to askbench.py")
    ap.add_argument("spec", help="experiment spec (.toml or .json)")
    ap.add_argument("--run-dir", help="run folder (default: ./askbench-runs/<name>-<spec hash>); rerun resumes")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the first 3 rendered calls; make no calls")
    ap.add_argument("--yes-calls", type=int, help="allow more than --cap calls; must equal the planned total")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help=f"refuse plans larger than this (default {DEFAULT_CAP})")
    ap.add_argument("--concurrency", type=int, default=None, help=f"parallel calls (default spec value or {DEFAULT_CONCURRENCY})")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per call")
    ap.add_argument("--effort", default=None, help="pass --effort to the subject (low..max)")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report from the run folder; make no calls")
    a = ap.parse_args(argv)

    try:
        spec = load_spec(a.spec)
    except (SpecError, tomllib.TOMLDecodeError, json.JSONDecodeError, OSError) as e:
        print(f"askbench: bad spec: {e}", file=sys.stderr)
        return 2
    effort = a.effort or spec.get("effort")
    conc = a.concurrency or spec.get("concurrency") or DEFAULT_CONCURRENCY
    run_dir = os.path.abspath(a.run_dir or os.path.join("askbench-runs", f"{spec['name']}-{spec['_sha']}"))
    calls_path = os.path.join(run_dir, "calls.jsonl")
    calls = plan_calls(spec)
    done = {r["id"] for r in read_records(calls_path) if r.get("ok")}
    todo = [c for c in calls if c["id"] not in done]

    if a.report_only:
        if not os.path.exists(calls_path):
            print(f"askbench: no calls.jsonl in {run_dir}", file=sys.stderr)
            return 2
        return _write_report(spec, calls_path, run_dir)

    total = print_plan(spec, todo, len(done))
    if a.dry_run:
        print("\n--- first 3 calls in run order (dry run: no calls made) ---")
        for c in calls[:3]:
            print(f"\n[{c['id']}]")
            print("command: " + " ".join(_shq(x) for x in claude_cmd(spec["model"], "<system below>", effort)) + "  < prompt on stdin")
            print(f"system prompt:\n{c['system'] or '(empty)'}")
            print(f"prompt:\n{c['prompt']}")
        print(f"\nrun folder would be: {run_dir}")
        return 0
    if not todo:
        print("nothing to do: every call is finished; rebuilding the report")
        return _write_report(spec, calls_path, run_dir)
    if total > a.cap and a.yes_calls != total:
        print(f"askbench: refusing: {total} calls exceeds the cap of {a.cap}. "
              f"Rerun with --yes-calls {total} to confirm.", file=sys.stderr)
        return 3
    if a.yes_calls is not None and a.yes_calls != total:
        print(f"askbench: --yes-calls {a.yes_calls} does not match the planned {total}", file=sys.stderr)
        return 3
    if shutil.which(os.environ.get("ASKBENCH_CLAUDE", "claude")) is None:
        print("askbench: `claude` not found on PATH", file=sys.stderr)
        return 2

    os.makedirs(run_dir, exist_ok=True)
    ext = ".json" if spec["_path"].endswith(".json") else ".toml"
    spec_copy = os.path.join(run_dir, "spec" + ext)
    if not os.path.exists(spec_copy):
        with open(spec_copy, "w") as f:
            f.write(spec["_raw"])
    plan = {"askbench_version": VERSION, "spec_sha": spec["_sha"], "seed": spec["seed"],
            "model": spec["model"], "effort": effort, "concurrency": conc,
            "cli_version": cli_version(), "flags": CLEAN_FLAGS, "env_passed": list(ENV_ALLOW),
            "judge_system": JUDGE_SYSTEM if spec.get("judge") else None,
            "order": [c["id"] for c in calls],
            "invocations": []}
    plan_path = os.path.join(run_dir, "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path) as f:
            old = json.load(f)
        plan["invocations"] = old.get("invocations", [])
    plan["invocations"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "argv": redact_argv(sys.argv[1:]), "todo": len(todo)})
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=1)

    workroot = tempfile.mkdtemp(prefix="askbench-")
    lock = threading.Lock()
    n_done = 0
    t0 = time.monotonic()
    try:
        with open(calls_path, "a") as out, cf.ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(run_one, spec, c, effort, a.timeout, workroot): c for c in todo}
            for fut in cf.as_completed(futs):
                c = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {"id": c["id"], "condition": c["condition"], "item": c["item"],
                           "sample": c["sample"], "ok": False, "error": repr(e), "outcome": None}
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n_done += 1
                    status = "ok" if rec.get("ok") else f"FAILED ({rec.get('exit_status', rec.get('error'))})"
                    print(f"  [{n_done}/{len(todo)}] {c['id']}: {status} outcome={rec.get('outcome')!r} "
                          f"{rec.get('duration_s', 0):.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; finished calls are saved, rerun the same command to resume", file=sys.stderr)
        return 130
    finally:
        shutil.rmtree(workroot, ignore_errors=True)
    print(f"finished {n_done} calls in {time.monotonic() - t0:.0f}s")
    return _write_report(spec, calls_path, run_dir)


def _write_report(spec, calls_path, run_dir) -> int:
    recs = read_records(calls_path)
    rep = build_report(spec, recs, run_dir)
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write(rep)
    print()
    print(rep)
    print(f"report written to {os.path.join(run_dir, 'report.md')}")
    return 0


def _shq(s):
    return s if s and re.fullmatch(r"[\w@%+=:,./-]+", s) else "'" + s.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    sys.exit(main())
