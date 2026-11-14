#!/usr/bin/env python3
"""lab - workbench for small interpretability experiments on small open models.

Stdlib only. The real work happens in the venv (~/.local/share/lab/venv,
override with LAB_VENV) with lablib on PYTHONPATH. `lab setup` builds the venv
with `uv` when there is one, else with `python3 -m venv` and pip (LAB_NO_UV=1
forces that). KEY=VALUE lines in ~/.config/lab/env are defaults for the
environment (a variable already set wins).

LAB_HOME puts everything lab keeps on disk in one folder: venv/ (the venv),
hf/ (the Hugging Face cache), cache/ (activations, fitted lenses, npedia) and
env (the settings file). `lab setup` marks that folder with a file .lab-home.
When LAB_HOME is not set and ~/projects/lab-env holds that marker, that folder
is used, so that programs which are not given the variable (agents in a
sandbox, for example) find the same venv, models and caches.

  lab setup [--no-models]   build the venv, install, fetch models, run check
  lab check                 PASS/FAIL lines; exit code 0 only if all pass
  lab run [--timeout S] SCRIPT [args]   run a script (default 600 s); what follows SCRIPT is the script's
  lab py [args]             the venv's python (lablib importable)
  lab models                known models: params, RAM, lens, cached
  lab fetch MODEL           download a model (+ its published lens)
  lab info                  device "auto" would pick and why; memory; versions
  lab new NAME              write a starter script NAME.py here
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(os.path.realpath(__file__)).parent
SHARED = "projects/lab-env"  # under the home folder; used as LAB_HOME once `lab setup` has marked it
MARKER = ".lab-home"  # the file `lab setup` writes into a lab home
VENV = Path(os.environ.get("LAB_VENV", Path.home() / ".local/share/lab/venv"))
PY = VENV / "bin" / "python"
GPT2 = "openai-community/gpt2"


def say(msg: str) -> None:
    print(f"[lab] {msg}", flush=True)


def load_settings(path: Path | None = None) -> list[str]:
    """Machine-wide defaults: KEY=VALUE lines in ~/.config/lab/env (LAB_ENV_FILE moves it).
    They never override a variable that is already set. Returns the keys it set."""
    path = path or Path(os.environ.get("LAB_ENV_FILE", Path.home() / ".config/lab/env"))
    done = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return done
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.isidentifier() and k not in os.environ:
            os.environ[k] = v
            done.append(k)
    return done


def lab_home() -> Path | None:
    """LAB_HOME if set, else ~/projects/lab-env if `lab setup` made it (it holds the marker file), else
    None (the separate default places). A folder that merely has that name is not used."""
    h = os.environ.get("LAB_HOME")
    if h:
        return Path(h).expanduser()
    shared = Path.home() / SHARED
    return shared if (shared / MARKER).is_file() else None


def mark_home() -> None:
    """Write the marker into the lab home, if there is one, so that later runs find it without the variable."""
    h = os.environ.get("LAB_HOME")
    if h:
        home = Path(h).expanduser()
        home.mkdir(parents=True, exist_ok=True)
        (home / MARKER).write_text("A lab home, made by `lab setup`.\n")


def resolve_paths() -> Path | None:
    """Read the settings, then fix where the venv, the Hugging Face cache and lab's caches live.
    A variable that is already set always wins. Returns the lab home, if there is one."""
    global VENV, PY
    load_settings()
    home = lab_home()
    if home:
        if (home / "env").is_file() and "LAB_ENV_FILE" not in os.environ:
            load_settings(home / "env")
        os.environ["LAB_HOME"] = str(home)
        os.environ.setdefault("LAB_VENV", str(home / "venv"))
        os.environ.setdefault("HF_HOME", str(home / "hf"))
        os.environ.setdefault("LAB_CACHE", str(home / "cache"))
    VENV = Path(os.environ.get("LAB_VENV", Path.home() / ".local/share/lab/venv")).expanduser()
    PY = VENV / "bin" / "python"
    return home


def queued() -> bool:
    return bool(shutil.which("heavy")) and not os.environ.get("LAB_NO_HEAVY")


@contextlib.contextmanager
def one_at_a_time():
    """Let one model-loading lab command run at a time (a file lock), so two runs do not load two models into the
    same memory limit. Waiting does not count against --timeout. LAB_NO_LOCK=1 skips it. Not used when a `heavy`
    job queue does the scheduling."""
    if queued() or os.environ.get("LAB_NO_LOCK") or os.environ.get("LAB_LOCK_HELD"):
        yield  # LAB_LOCK_HELD: this is a lab command started by a script that a locked lab command is running
        return
    import fcntl
    d = Path(os.environ.get("LAB_CACHE", Path.home() / ".cache" / "lab")).expanduser()
    try:
        d.mkdir(parents=True, exist_ok=True)
        f = open(d / "run.lock", "a")
    except OSError:  # nowhere to keep a lock: run without one
        yield
        return
    with f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            say("another lab run is using the model memory; waiting for it to finish")
            t0 = time.time()
            fcntl.flock(f, fcntl.LOCK_EX)
            say(f"waited {time.time() - t0:.0f} s")
        os.environ["LAB_LOCK_HELD"] = "1"
        try:
            yield
        finally:
            del os.environ["LAB_LOCK_HELD"]  # closing the file releases the lock


def env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(HERE) + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    return e


def wrap(cmd: list[str], timeout: int | None = None, quick: bool = False) -> list[str]:
    """Put a command under `timeout`, and under a job queue named `heavy` when the machine has one on the PATH
    (optional; most machines do not), with the timeout inside so queue time does not count."""
    inner = (["timeout", "--signal=TERM", "--kill-after=10", str(timeout)] if timeout and
             shutil.which("timeout") else []) + cmd
    if queued():
        return ["heavy"] + (["--quick"] if quick else []) + inner
    return inner


def need_venv() -> None:
    if not PY.exists():
        sys.exit(f"[lab] no venv at {VENV}; run: lab setup")


def vpy(code: str, timeout: int | None = None, quick: bool = False, heavy: bool = True) -> int:
    cmd = [str(PY), "-c", code]
    if not heavy:
        return subprocess.call(cmd, env=env())
    with one_at_a_time():
        return subprocess.call(wrap(cmd, timeout, quick), env=env())


def find_uv() -> str | None:
    return shutil.which("uv") or next(
        (str(p) for p in [Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv"]
         if p.exists()), None)


# --------------------------------------------------------------------------
def cmd_setup(args: list[str]) -> int:
    t0 = time.time()
    uv = None if os.environ.get("LAB_NO_UV") else find_uv()
    mark_home()
    if PY.exists():
        say(f"venv exists at {VENV}")
    else:
        VENV.parent.mkdir(parents=True, exist_ok=True)
        if uv:
            say(f"creating venv (Python 3.12) at {VENV}")
            r = subprocess.call([uv, "venv", "--python", "3.12", str(VENV)])
        else:
            # No uv on this machine: the system Python and pip do the same job, a little slower.
            if sys.version_info < (3, 10):
                say(f"Python {sys.version.split()[0]} is too old (need 3.10 or newer), and there is no uv")
                return 1
            say(f"creating venv (Python {sys.version.split()[0]}, no uv found) at {VENV}")
            r = subprocess.call([sys.executable, "-m", "venv", str(VENV)])
        if r:
            say("could not create the venv" + ("" if uv else " (on Debian/Ubuntu: apt install python3-venv)"))
            return r
    say("installing requirements.txt (first time: a few minutes; torch is large)")
    req = str(HERE / "requirements.txt")
    if uv:
        r = subprocess.call([uv, "pip", "install", "--python", str(PY), "-r", req])
    elif subprocess.call([str(PY), "-m", "pip", "--version"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL) == 0:
        r = subprocess.call([str(PY), "-m", "pip", "install", "--disable-pip-version-check", "-r", req])
    else:
        say(f"the venv at {VENV} has no pip and there is no uv; delete it and run lab setup again")
        return 1
    if r:
        say("pip install failed")
        return r
    if "--no-models" not in args:
        for m in ("Qwen/Qwen3.5-0.8B", GPT2):
            say(f"fetching {m} and its lens")
            r = vpy(f"from lablib.models import fetch; fetch({m!r})", heavy=False)
            if r:
                say(f"fetch {m} failed")
                return r
    say(f"setup took {time.time() - t0:.0f} s; running check")
    return cmd_check([])


CHECK_CODE = r'''
import sys, time, os
ok = True
def line(passed, what, detail=""):
    global ok
    ok = ok and passed
    print(("PASS " if passed else "FAIL ") + what + (f"  ({detail})" if detail else ""), flush=True)
try:
    import torch, transformers, jlens, sklearn, numpy, scipy, lablib
    line(True, "imports", f"torch {torch.__version__}, transformers {transformers.__version__}")
except Exception as e:
    line(False, "imports", repr(e)); sys.exit(1)
from lablib import models
dev, why = models.pick_device(models.DEFAULT)
line(True, "device auto", why)
for m in (models.DEFAULT, "openai-community/gpt2"):
    line(models.is_cached(m), f"model cached: {m}", "" if models.is_cached(m) else "run: lab fetch " + m)
    line(models.lens_cached(m), f"lens cached: {m}")
skip = os.environ.get("LAB_CHECK_NO_TESTS")
if not skip and models.is_cached("openai-community/gpt2"):
    import pytest
    t = time.time()
    r = pytest.main(["-q", "-x", "-m", "quick", "-p", "no:cacheprovider",
                     os.path.join(os.environ["LAB_HERE"], "tests", "test_lablib.py")])
    line(r == 0, "quick tests on gpt2", f"pytest exit {int(r)}, {time.time()-t:.0f} s")
print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
'''


def cmd_check(args: list[str]) -> int:
    if not PY.exists():
        print(f"FAIL venv  (no {PY}; run: lab setup)")
        return 1
    print(f"PASS venv  ({VENV})", flush=True)
    os.environ["LAB_HERE"] = str(HERE)
    return vpy(CHECK_CODE, timeout=290, quick=True)


RUN_USAGE = ("usage: lab run [--timeout S] SCRIPT [args]\n"
             "  lab's own options go before SCRIPT; everything after SCRIPT is passed to the script unchanged")


def parse_run(args: list[str]) -> tuple[int, list[str]]:
    """(timeout, [script, its args...]). Only options before the script are lab's."""
    timeout, args = 600, list(args)
    while args and args[0].startswith("-"):
        opt = args.pop(0)
        if opt == "--":
            break
        if opt == "--timeout":
            if not args:
                sys.exit(f"[lab] run: --timeout needs a number of seconds\n{RUN_USAGE}")
            val = args.pop(0)
        elif opt.startswith("--timeout="):
            val = opt.split("=", 1)[1]
        else:
            sys.exit(f"[lab] run: unknown option {opt}\n{RUN_USAGE}")
        if not val.isdigit() or int(val) < 1:
            sys.exit(f"[lab] run: --timeout needs a whole number of seconds, got {val!r}\n{RUN_USAGE}")
        timeout = int(val)
    if not args:
        sys.exit(RUN_USAGE)
    return timeout, args


def cmd_run(args: list[str]) -> int:
    timeout, args = parse_run(args)
    need_venv()
    if "--timeout" in args[1:] or any(a.startswith("--timeout=") for a in args[1:]):
        say("note: --timeout after the script goes to the script; put it before the script to set lab's limit")
    cmd = wrap([str(PY), args[0]] + args[1:], timeout)
    t0 = time.time()
    with one_at_a_time():
        r = subprocess.call(cmd, env=env())
    if r == 124:
        say(f"TIMEOUT after {timeout} s (raise it: lab run --timeout S SCRIPT)")
    say(f"exit {r}, {time.time() - t0:.0f} s wall (including any queue wait)")
    return r


def cmd_py(args: list[str]) -> int:
    need_venv()
    os.execve(str(PY), [str(PY)] + args, env())


def cmd_models(args: list[str]) -> int:
    need_venv()
    return vpy(r'''
from lablib import models
from lablib.stats import print_table
rows = []
for name, s in models.KNOWN.items():
    rows.append([name, f"{s['params']/1e6:.0f}M", f"{models.ram_needed_bytes(name)/1e9:.1f} GB",
                 "published" if s.get("lens") else "fit", "yes" if models.is_cached(name) else "no",
                 "yes" if models.lens_cached(name) else "no", s.get("note", "")])
print_table(rows, ["model", "params", "RAM fp32", "lens", "model cached", "lens cached", "note"])
print("default:", models.DEFAULT, "| aliases:", ", ".join(models.ALIASES))
print("other HF causal LMs also load; their lens is fitted on a small sample.")
''', heavy=False)


def cmd_fetch(args: list[str]) -> int:
    need_venv()
    if not args:
        sys.exit("usage: lab fetch MODEL")
    return vpy(f"from lablib.models import fetch; fetch({args[0]!r})", heavy=False)


def cmd_info(args: list[str]) -> int:
    need_venv()
    return vpy(r'''
import os
from lablib import models
from lablib.report import versions
dev, why = models.pick_device(models.DEFAULT)
print(f"device auto for {models.DEFAULT}: {dev}  ({why})")
try:
    mem = dict(l.split(":", 1) for l in open("/proc/meminfo"))
    print(f"RAM available: {int(mem['MemAvailable'].split()[0])/1e6:.1f} GB")
except Exception:
    pass
free, how = models.gpu_free_bytes()
print("GPU free:", f"{free/1e9:.2f} GB ({how})" if free is not None else how)
import torch
print("torch threads:", torch.get_num_threads(), "(LAB_THREADS overrides)")
print("versions:", versions())
print("venv:", os.environ.get("VIRTUAL_ENV") or __import__("sys").prefix)
print("lab home:", os.environ.get("LAB_HOME") or "not set (venv, Hugging Face cache and lab cache are in their default places)")
print("Hugging Face cache:", os.environ.get("HF_HOME") or "~/.cache/huggingface")
print("lab cache:", os.environ.get("LAB_CACHE") or "~/.cache/lab")
''', heavy=False)


TEMPLATE = '''"""{name}: describe the question in one sentence.

Run: lab run {name}.py
"""
import lablib
from lablib import acts, data, probe, report, stats

SEED = 0
m = lablib.load()  # default model; lablib.load("gpt2") for a fast try
ds = data.sentiment()
layers = [m.n_layers // 4, m.n_layers // 2, 3 * m.n_layers // 4]
A = acts.get_acts(m, ds["prompts"], layers, position="last")
rows = probe.sweep_layers(A, ds["labels"], ds["groups"], seed=SEED)
stats.print_table(rows, probe.SWEEP_COLS)
report.run_dir("{name}").save({{"rows": rows}}, rows=rows, cols=probe.SWEEP_COLS,
                             model=m, seed=SEED)
'''


def cmd_new(args: list[str]) -> int:
    if not args:
        sys.exit("usage: lab new NAME")
    name = args[0].removesuffix(".py")
    p = Path.cwd() / f"{name}.py"
    if p.exists():
        sys.exit(f"[lab] {p} exists; not overwriting")
    p.write_text(TEMPLATE.format(name=name))
    say(f"wrote {p}; run it with: lab run {p.name}")
    return 0


CMDS = {"setup": cmd_setup, "check": cmd_check, "run": cmd_run, "py": cmd_py,
        "models": cmd_models, "fetch": cmd_fetch, "info": cmd_info, "new": cmd_new}


def main() -> int:
    resolve_paths()
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help") or sys.argv[1] not in CMDS:
        print(__doc__.strip())
        return 0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help") else 2
    return CMDS[sys.argv[1]](sys.argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main())
