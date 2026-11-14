"""Save an experiment's results in one directory.

    run = report.run_dir("probe-sentiment")      # creates the directory
    run.save(results, rows=rows, cols=cols, model=m, seed=0, fig=fig)

Directory: $LAB_RUN_DIR if set (used as-is, created if needed), otherwise
./lab-runs/<UTC timestamp>-<name>/. Contents:
  results.json  your results + metadata (model, seed, versions, argv, time)
                (in argv, the value of an option named like a key, token, secret or password is saved as ***)
  results.md    markdown table (if rows given) + metadata
  figure.png    (if fig given; a matplotlib Figure)
  script.py     copy of the running script (sys.argv[0]), if it is a file
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path


_SECRET_NAME = re.compile(r"(^|[-_])(api)?key$|(^|[-_])token$|token[-_]?(id|secret)$|secret|passw|credential",
                          re.I)


def redact_argv(argv) -> list:
    """argv with the value of any option whose name looks like a key, token, secret or password replaced by ***
    (`--api-key V`, `--api-key=V`, `HF_TOKEN=V`). Counts of tokens (`--max-new-tokens 20`) are left alone."""
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


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "jlens", "numpy", "sklearn", "lablib"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            out[mod] = None
    return out


def _jsonable(x):
    try:
        import numpy as np
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


class Run:
    def __init__(self, path: Path, name: str):
        self.path, self.name = path, name

    def __repr__(self):
        return f"Run({self.path})"

    def save(self, results, rows=None, cols=None, model=None, seed=None, fig=None,
             notes: str = "") -> Path:
        """Write results.json, results.md, optional figure.png, script.py."""
        from .stats import table_str

        mname = getattr(model, "name", model)
        meta = {"name": self.name, "model": mname,
                "device": getattr(model, "device", None),
                "dtype": str(getattr(model, "dtype", "")) or None, "seed": seed,
                "versions": versions(), "argv": redact_argv(sys.argv),
                "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        (self.path / "results.json").write_text(
            json.dumps(_jsonable({"meta": meta, "results": results}), indent=2))
        md = [f"# {self.name}", ""]
        if notes:
            md += [notes, ""]
        if rows is not None:
            if cols is None:
                cols = list(rows[0]) if rows and isinstance(rows[0], dict) else []
            md += [table_str(rows, cols), ""]
        md += ["## meta", "", "```", json.dumps(_jsonable(meta), indent=2), "```", ""]
        (self.path / "results.md").write_text("\n".join(md))
        if fig is not None:
            fig.savefig(self.path / "figure.png", dpi=120, bbox_inches="tight")
        script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
        if script and script.is_file() and script.suffix == ".py":
            shutil.copy(script, self.path / "script.py")
        print(f"[lab] saved results to {self.path}", flush=True)
        return self.path


def run_dir(name: str) -> Run:
    env = os.environ.get("LAB_RUN_DIR")
    if env:
        p = Path(env)
    else:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "run"
        p = Path.cwd() / "lab-runs" / f"{stamp}-{safe}"
    p.mkdir(parents=True, exist_ok=True)
    return Run(p, name)
