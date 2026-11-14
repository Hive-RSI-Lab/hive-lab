"""Run a local lab experiment script on a Modal GPU and copy its run folder back.

NOT YET RUN AGAINST A REAL ACCOUNT (see modal/README.md).

Usage (needs `pip install modal` and a signed-in Modal account):

    modal run modal/run_remote.py --script examples/my_exp.py
    modal run modal/run_remote.py --script examples/my_exp.py --gpu L4 --timeout-min 40 \
        --args "--layers 4 --n 200" --out runs/my_exp-remote

What it does:
- Builds an image (Debian slim, Python 3.12) that installs the module's requirements.txt
  if that file exists, and adds lablib/ so `import lablib` works.
- Runs one function on the chosen GPU (default T4; also L4, A10, A100, A100-40GB, A100-80GB)
  with a timeout (default 20 minutes), and a persistent Volume "lab-hf-cache" mounted as the
  Hugging Face cache (HF_HOME), so model downloads are reused between runs.
- Sends the script's bytes, runs `python SCRIPT ARGS` remotely with LAB_RUN_DIR set to an
  empty folder, then sends that folder back as a .tar.gz and unpacks it into --out locally.

Offline check without any Modal account or token:

    python modal/run_remote.py --check
"""
from __future__ import annotations

import io
import os
import shlex
import sys
import tarfile
import time

import modal

HERE = os.path.dirname(os.path.realpath(__file__))
MODULE_DIR = os.path.dirname(HERE)  # /.../modules/lab
REQUIREMENTS = os.path.join(MODULE_DIR, "requirements.txt")
LABLIB = os.path.join(MODULE_DIR, "lablib")

REMOTE_ROOT = "/root/lab"
REMOTE_RUN_DIR = "/root/lab_run"
HF_CACHE = "/cache/hf"
VOLUME_NAME = "lab-hf-cache"
GPU_CHOICES = ("T4", "L4", "A10", "A100", "A100-40GB", "A100-80GB")
DEFAULT_GPU = "T4"
DEFAULT_TIMEOUT_MIN = 20


def build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.12").apt_install("git")  # requirements.txt has a git+https line
    if os.path.isfile(REQUIREMENTS):
        image = image.pip_install_from_requirements(REQUIREMENTS)
    image = image.env({"PYTHONPATH": REMOTE_ROOT, "HF_HOME": HF_CACHE, "PYTHONUNBUFFERED": "1"})
    if os.path.isdir(LABLIB):
        image = image.add_local_dir(LABLIB, f"{REMOTE_ROOT}/lablib",
                                    ignore=["**/__pycache__", "**/*.pyc"])
    return image


app = modal.App("lab-run-remote")
image = build_image()
hf_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=image, gpu=DEFAULT_GPU, timeout=DEFAULT_TIMEOUT_MIN * 60,
              volumes={HF_CACHE: hf_volume})
def run_experiment(script_name: str, script_bytes: bytes, args: list[str]) -> dict:
    """Runs the script remotely; returns exit code, timing and the run folder as tar.gz bytes."""
    import shutil
    import subprocess

    shutil.rmtree(REMOTE_RUN_DIR, ignore_errors=True)
    os.makedirs(REMOTE_RUN_DIR)
    script_path = os.path.join(REMOTE_ROOT, os.path.basename(script_name))
    with open(script_path, "wb") as fh:
        fh.write(script_bytes)
    env = dict(os.environ, LAB_RUN_DIR=REMOTE_RUN_DIR)
    t0 = time.time()
    proc = subprocess.run([sys.executable, script_path, *args], cwd=REMOTE_ROOT, env=env)
    elapsed = time.time() - t0
    try:
        hf_volume.commit()  # keep downloaded models for the next run
    except Exception as e:  # noqa: BLE001 - a failed commit must not lose the results
        print(f"warning: volume commit failed: {e}", file=sys.stderr)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(REMOTE_RUN_DIR, arcname=".")
    return {"returncode": proc.returncode, "seconds": round(elapsed, 1), "tar": buf.getvalue()}


def safe_extract(tar_bytes: bytes, out_dir: str) -> list[str]:
    """Unpacks only regular files and folders inside out_dir (no links, no absolute paths)."""
    os.makedirs(out_dir, exist_ok=True)
    names = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            target = os.path.realpath(os.path.join(out_dir, m.name))
            if not (m.isfile() or m.isdir()):
                continue
            if not (target == os.path.realpath(out_dir) or
                    target.startswith(os.path.realpath(out_dir) + os.sep)):
                continue
            members.append(m)
            if m.isfile():
                names.append(m.name)
        tar.extractall(out_dir, members=members, filter="data")
    return names


def check_args(script: str, gpu: str, timeout_min: int) -> None:
    if not os.path.isfile(script):
        raise SystemExit(f"run_remote: script not found: {script}")
    if gpu not in GPU_CHOICES:
        raise SystemExit(f"run_remote: --gpu must be one of {', '.join(GPU_CHOICES)}")
    if not 1 <= timeout_min <= 24 * 60:
        raise SystemExit("run_remote: --timeout-min must be between 1 and 1440")


@app.local_entrypoint()
def main(script: str, gpu: str = DEFAULT_GPU, timeout_min: int = DEFAULT_TIMEOUT_MIN,
         args: str = "", out: str = ""):
    check_args(script, gpu, timeout_min)
    stem = os.path.splitext(os.path.basename(script))[0]
    out_dir = out or os.path.join("runs", f"{stem}-remote-{time.strftime('%Y%m%d-%H%M%S')}")
    with open(script, "rb") as fh:
        data = fh.read()
    print(f"run_remote: {script} on {gpu}, timeout {timeout_min} min -> {out_dir}")
    fn = run_experiment.with_options(gpu=gpu, timeout=timeout_min * 60)
    result = fn.remote(os.path.basename(script), data, shlex.split(args))
    files = safe_extract(result["tar"], out_dir)
    print(f"run_remote: exit code {result['returncode']}, {result['seconds']} s on the GPU, "
          f"{len(files)} files copied to {out_dir}")
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])


def offline_check() -> int:
    """Validates the app definition locally; makes no network call and needs no token."""
    print(f"modal {modal.__version__}")
    print(f"app: {app.name}; function: run_experiment")
    print(f"requirements.txt: {'found, installed into the image' if os.path.isfile(REQUIREMENTS) else 'not found'}")
    print(f"lablib/: {'found, added at ' + REMOTE_ROOT + '/lablib' if os.path.isdir(LABLIB) else 'not found'}")
    print(f"default GPU {DEFAULT_GPU}, timeout {DEFAULT_TIMEOUT_MIN} min, volume {VOLUME_NAME} at {HF_CACHE}")
    for g in GPU_CHOICES:  # with_options builds the variant locally only
        run_experiment.with_options(gpu=g, timeout=60)
    print("with_options OK for: " + ", ".join(GPU_CHOICES))
    # round-trip of the copy-back format, including a hostile member that must be skipped
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in (("./result.json", b"{}"), ("./sub/log.txt", b"ok"), ("../escape.txt", b"x")):
            ti = tarfile.TarInfo(name)
            ti.size = len(body)
            tar.addfile(ti, io.BytesIO(body))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        got = sorted(safe_extract(buf.getvalue(), os.path.join(d, "out")))
        assert got == ["./result.json", "./sub/log.txt"], got
        assert not os.path.exists(os.path.join(d, "escape.txt"))
    print("copy-back extraction OK (path escape skipped)")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        sys.exit(offline_check())
    print(__doc__)
    sys.exit(2)
