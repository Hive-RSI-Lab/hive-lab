"""Tests for lab.py, the launcher (stdlib only; no model is loaded).

Run: lab py -m pytest tests/test_launcher.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lab  # noqa: E402


def test_parse_run_options_before_the_script_are_labs():
    assert lab.parse_run(["x.py"]) == (600, ["x.py"])
    assert lab.parse_run(["--timeout", "30", "x.py"]) == (30, ["x.py"])
    assert lab.parse_run(["--timeout=45", "x.py", "a", "b"]) == (45, ["x.py", "a", "b"])
    # what follows the script belongs to the script, --timeout included
    assert lab.parse_run(["--timeout", "30", "x.py", "--timeout", "5", "--model", "gpt2"]) == (
        30, ["x.py", "--timeout", "5", "--model", "gpt2"])
    assert lab.parse_run(["x.py", "--timeout", "5"]) == (600, ["x.py", "--timeout", "5"])
    assert lab.parse_run(["--", "-odd-name.py", "--k", "3"]) == (600, ["-odd-name.py", "--k", "3"])
    assert lab.parse_run(["--timeout", "30", "--timeout", "40", "x.py"]) == (40, ["x.py"])


@pytest.mark.parametrize("args", [
    [],                            # no script
    ["--timeout"],                 # no value
    ["--timeout", "30"],           # a value but no script
    ["--timeout", "abc", "x.py"],
    ["--timeout", "0", "x.py"],
    ["--timeout", "-5", "x.py"],
    ["--timeout=", "x.py"],
    ["--timeout=1.5", "x.py"],
    ["--bogus", "x.py"],           # an unknown option is not passed on silently
])
def test_parse_run_rejects(args):
    with pytest.raises(SystemExit) as e:
        lab.parse_run(args)
    assert "usage: lab run" in str(e.value)  # the message, not a traceback


def test_wrap_puts_the_timeout_inside_and_the_queue_outside(monkeypatch):
    monkeypatch.setattr(lab.shutil, "which", lambda name: "/usr/bin/" + name if name == "timeout" else None)
    monkeypatch.delenv("LAB_NO_HEAVY", raising=False)
    assert lab.wrap(["py", "x.py"], 30) == ["timeout", "--signal=TERM", "--kill-after=10", "30", "py", "x.py"]
    assert lab.wrap(["py", "x.py"]) == ["py", "x.py"]
    # a machine with a `heavy` job queue: the queue goes outside, so waiting is not timed
    monkeypatch.setattr(lab.shutil, "which", lambda name: "/usr/bin/" + name)
    assert lab.wrap(["py", "x.py"], 30, quick=True)[:3] == ["heavy", "--quick", "timeout"]
    monkeypatch.setenv("LAB_NO_HEAVY", "1")
    assert lab.wrap(["py", "x.py"], 30)[0] == "timeout"


def test_settings_file_never_overrides(tmp_path, monkeypatch):
    f = tmp_path / "env"
    f.write_text("# a comment\nLAB_TEST_A=from-file\nLAB_TEST_B = \"quoted\"\nnot a setting\n1BAD=x\n")
    monkeypatch.setenv("LAB_TEST_A", "already-set")
    monkeypatch.delenv("LAB_TEST_B", raising=False)
    try:
        assert lab.load_settings(f) == ["LAB_TEST_B"]
        assert os.environ["LAB_TEST_A"] == "already-set" and os.environ["LAB_TEST_B"] == "quoted"
    finally:
        os.environ.pop("LAB_TEST_B", None)
    assert lab.load_settings(tmp_path / "missing") == []


def test_lab_home_puts_everything_in_one_folder(tmp_path, monkeypatch):
    for k in ("LAB_HOME", "LAB_VENV", "HF_HOME", "LAB_CACHE", "LAB_ENV_FILE"):
        monkeypatch.setenv(k, "x")  # recorded, so that what resolve_paths sets is undone after the test
        monkeypatch.delenv(k)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(lab, "VENV", lab.VENV)
    monkeypatch.setattr(lab, "PY", lab.PY)
    assert lab.resolve_paths() is None  # nothing set, no shared folder: the default places
    assert lab.VENV == tmp_path / ".local/share/lab/venv"
    shared = tmp_path / "projects" / "lab-env"
    shared.mkdir(parents=True)
    (shared / "env").write_text("LAB_THREADS_TEST=4\n")
    assert lab.resolve_paths() is None  # a folder that merely has the name is not a lab home
    assert "LAB_THREADS_TEST" not in os.environ and lab.VENV == tmp_path / ".local/share/lab/venv"
    monkeypatch.setenv("LAB_HOME", str(shared))
    lab.mark_home()  # what `lab setup` does first
    monkeypatch.delenv("LAB_HOME")
    assert (shared / lab.MARKER).is_file()
    try:
        assert lab.resolve_paths() == shared  # marked by setup: found without any variable
        assert lab.VENV == shared / "venv" and lab.PY == shared / "venv" / "bin" / "python"
        assert os.environ["HF_HOME"] == str(shared / "hf")
        assert os.environ["LAB_CACHE"] == str(shared / "cache")
        assert os.environ["LAB_THREADS_TEST"] == "4"
    finally:
        os.environ.pop("LAB_THREADS_TEST", None)
    monkeypatch.setenv("LAB_VENV", str(tmp_path / "elsewhere"))
    lab.resolve_paths()
    assert lab.VENV == tmp_path / "elsewhere"  # a variable that is set wins


# hive-lab #1: --check warns when a proxy is set and Modal's proxy packages are missing.
# Kept in this file so CI runs it without a workflow change; modal is not imported.
import importlib.util  # noqa: E402
import pathlib  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "proxycheck", pathlib.Path(__file__).resolve().parent.parent / "modal" / "proxycheck.py")
proxycheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxycheck)
warn = proxycheck.proxy_warning


def test_no_proxy_no_warning():
    assert warn({}, lambda m: False) is None


def test_proxy_with_packages_no_warning():
    assert warn({"HTTPS_PROXY": "http://127.0.0.1:3128"}, lambda m: True) is None


def test_proxy_missing_packages_warns_with_the_fix():
    w = warn({"https_proxy": "http://127.0.0.1:3128"}, lambda m: m == "aiohttp_socks")
    assert w and "https_proxy" in w and "python_socks" in w and "aiohttp_socks" not in w.split("but")[1].split("is not")[0]
    assert "modal[api-proxy-support]" in w


def test_empty_proxy_variable_ignored():
    assert warn({"HTTPS_PROXY": ""}, lambda m: False) is None
