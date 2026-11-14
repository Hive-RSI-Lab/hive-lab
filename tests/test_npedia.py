"""Offline tests for npedia.py. No network: a fake opener serves made-up responses with the same
shape (keys and types) as Neuronpedia's API responses on 2026-11-14. The model and source ids are
real; every explanation, text, token list, number and share id below is invented."""
import io
import json
import os
import sys
import urllib.error

import pytest

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import npedia  # noqa: E402


def _expl(text):
    return {"explanations": [{"description": text}]}


def _lens_row(position, token, runs):
    """One token's lens result: `runs` is [(top token, how many layers in a row)], 64 layers in all."""
    tops = [[tok] for tok, n in runs for _ in range(n)]
    assert len(tops) == 64
    return {"kind": "token", "position": position, "token": token, "is_generated": False,
            "results": [{"type": "JACOBIAN_LENS", "top_tokens": tops}]}


_RESOURCES_HTML = (
    '<table><thead><tr><th class="px-4 py-2 text-left">Model ID</th><th class="px-4 py-2 text-left">Source ID</th>'
    '<th class="px-4 py-2 text-left">Inference Enabled<!-- --> <span class="inline-block"><svg width="24" height="24">'
    '<circle cx="12" cy="12" r="10"></circle></svg></span></th></tr></thead><tbody>'
    '<tr class="hover:bg-sky-100 bg-slate-50 "><td class="w-1/3 px-4 py-1"><a target="_blank" '
    'href="https://www.neuronpedia.org/circuitgpt-python" class="font-mono">circuitgpt-python</a></td>'
    '<td class="w-1/3 px-4 py-1"><a target="_blank" href="https://www.neuronpedia.org/circuitgpt-python/0-mlp" '
    'class="font-mono">0-mlp</a></td><td class="w-1/3 px-4 py-1"><div class="flex flex-row">❌<!-- --> '
    '<button class="inline-flex">Request</button></div></td></tr>'
    '<tr class="hover:bg-sky-100 bg-white "><td class="w-1/3 px-4 py-1"><a target="_blank" '
    'href="https://www.neuronpedia.org/gpt2-small" class="font-mono">gpt2-small</a></td>'
    '<td class="w-1/3 px-4 py-1"><a target="_blank" href="https://www.neuronpedia.org/gpt2-small/6-res-jb" '
    'class="font-mono">6-res-jb</a></td><td class="w-1/3 px-4 py-1">✅</td></tr></tbody></table>')

SAMPLES = {
    "feature": {
        "modelId": "gpt2-small", "layer": "0-res-jb", "index": "14057",
        "maxActApprox": 2.25, "frac_nonzero": 0.0005,
        "explanations": [
            {"description": "words about rain and wet weather", "explanationModelName": "explainer-a",
             "typeName": "oai_token-act-pair"},
            {"description": "the word \"rain\" in a weather report", "explanationModelName": "explainer-b",
             "typeName": "oai_token-act-pair"}],
        "pos_str": [" rain", " storm", " umbrella"], "pos_values": [3.2, 2.85, 2.84],
        "neg_str": ["desert", " desert", " dry"], "neg_values": [-0.88, -0.83, -0.82],
        "activations": [
            {"tokens": [" A", " day", " of", " heavy", " rain", ",", " then", " sun"],
             "values": [0, 0, 0, 0, 2.25, 0, 0, 0], "maxValue": 2.25, "maxValueTokenIndex": 4},
            {"tokens": [" We", " walked", " in", " the", " rain", " all", " the", " way"],
             "values": [0, 0, 0, 0, 2.2, 0, 0, 0], "maxValue": 2.2, "maxValueTokenIndex": 4}]},
    "search_all": {
        "request": {"query": "dogs", "offset": 0},
        "results": [
            {"modelId": "qwen3-4b", "layer": "8-transcoder-hp", "index": "1001",
             "description": "words about dogs", "cosine_similarity": 0.61},
            {"modelId": "qwen3-4b", "layer": "8-transcoder-hp", "index": "1002",
             "description": "pets and other animals at home", "cosine_similarity": 0.55},
            {"modelId": "qwen3-4b", "layer": "9-transcoder-hp", "index": "1003",
             "description": "the word \"dog\" in idioms", "cosine_similarity": 0.52}],
        "resultsCount": 20, "hasMore": True, "nextOffset": 20},
    "topk": {
        "source": "6-res-jb",
        "results": [
            {"position": 1, "token": "The", "isSpecial": False, "topFeatures": [
                {"activationValue": 32.5, "featureIndex": 1201, "feature": _expl("the word \"The\" starting a sentence")},
                {"activationValue": 24.5, "featureIndex": 1202, "feature": _expl("the first word of a text")}]},
            {"position": 2, "token": " cat", "isSpecial": False, "topFeatures": [
                {"activationValue": 49.5, "featureIndex": 1203, "feature": _expl("the word \"cat\"")},
                {"activationValue": 16.25, "featureIndex": 1204, "feature": _expl("cats and other pets")}]}]},
    "search_all_layers": {
        "tokens": ["<|endoftext|>", "The", " cat", " sat"],
        "result": [
            {"layer": "11-res-jb", "index": "301", "maxValue": 93.5, "maxValueIndex": 2,
             "neuron": _expl("words about animals")},
            {"layer": "12-res-jb", "index": "302", "maxValue": 83.5, "maxValueIndex": 2,
             "neuron": _expl("phrases with the word \"cat\"")}]},
    "act": {
        "tokens": ["The", " cat", " sat", " on", " the", " mat"], "values": [0, 3.5, 0, 0, 0, 1.25],
        "maxValue": 3.5, "minValue": 0, "modelId": "gpt2-small", "layer": "6-res-jb", "index": "100",
        "creatorId": "", "dataIndex": None, "dataSource": "Neuronpedia", "maxValueTokenIndex": 1,
        "createdAt": "2026-01-01T00:00:00.000Z"},
    "steer": {
        "STEERED": " the rain will fall and the rain will fall again", "DEFAULT": " the weather will be fine tomorrow",
        "steeredLogProbs": None, "defaultLogProbs": None, "id": "exampleid0000000000000000",
        "shareUrl": "https://www.neuronpedia.org/steer/exampleid0000000000000000", "limit": "239"},
    "lens": {
        "meta": {"kind": "meta", "model": "Qwen/Qwen3.6-27B", "types": ["JACOBIAN_LENS"],
                 "layers_by_type": {"JACOBIAN_LENS": list(range(64))}, "top_n": 3, "prompt_len": 5,
                 "num_completion_tokens": 1, "temperature": 0, "prepend_bos": True, "reuse_len": 0,
                 "oracle_layers": [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]},
        "tokens": [
            _lens_row(3, " France", [("e", 1), ("a", 3), (" France", 47), (" is", 7), (" Paris", 1), (" is", 5)]),
            _lens_row(4, " is", [(" The", 2), (" therefore", 10), (" only", 13), (" what", 15), (" cities", 5),
                                 (" Paris", 19)])],
        "done": {"kind": "done", "seq_len": 6, "prompt_len": 5, "vocab_size": 248320, "completion": " Paris"}},
    "resources_html": _RESOURCES_HTML,
}


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeOpener:
    """Maps URL path -> sample; records requests."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, req, timeout=None):
        path = "/" + req.full_url.split("://", 1)[1].split("/", 1)[1]
        body = json.loads(req.data) if req.data else None
        self.requests.append({"path": path, "body": body, "headers": dict(req.header_items()),
                              "method": req.get_method(), "timeout": timeout})
        r = self.routes.get(path)
        if r is None:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b'{"message":"nf"}'))
        if isinstance(r, Exception):
            raise r
        data = r if isinstance(r, str) else json.dumps(r)
        return FakeResp(data.encode())


ROUTES = {
    "/available-resources": SAMPLES["resources_html"],
    "/api/feature/gpt2-small/0-res-jb/14057": SAMPLES["feature"],
    "/api/explanation/search-all": SAMPLES["search_all"],
    "/api/explanation/search": SAMPLES["search_all"],
    "/api/search-topk-by-token": SAMPLES["topk"],
    "/api/search-all": SAMPLES["search_all_layers"],
    "/api/activation/new": SAMPLES["act"],
    "/api/steer": SAMPLES["steer"],
    "/api/lens/prompt": SAMPLES["lens"],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv(npedia.KEY_ENV, raising=False)
    monkeypatch.setattr(npedia, "MIN_INTERVAL", 0.0)
    opener = FakeOpener(dict(ROUTES))

    def make(use_cache=True, base="https://neuronpedia.test"):
        return npedia.Client(base=base, cache_dir=str(tmp_path / "cache"),
                             use_cache=use_cache, opener=opener)
    return opener, make


def run(capsys, client, *argv):
    rc = npedia.main(list(argv), client=client)
    out, err = capsys.readouterr()
    return rc, out, err


def test_models_and_sources(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "models")
    assert rc == 0
    assert "circuitgpt-python" in out and "gpt2-small" in out
    rc, out, _ = run(capsys, make(), "sources", "gpt2-small")
    assert rc == 0 and "6-res-jb" in out and "yes" in out
    rc, out, _ = run(capsys, make(), "sources", "circuitgpt-python")
    assert "0-mlp" in out and "no" in out
    rc, _, err = run(capsys, make(), "sources", "no-such-model")
    assert rc == 1 and "no sources" in err


def test_parse_resources_flags():
    rows = npedia.parse_resources(SAMPLES["resources_html"])
    assert {"model": "gpt2-small", "source": "6-res-jb", "inference": True} in rows
    assert {"model": "circuitgpt-python", "source": "0-mlp", "inference": False} in rows


def test_feature_table(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "feature", "gpt2-small", "0-res-jb", "14057", "--texts", "2", "--logits", "3")
    assert rc == 0
    assert "wet weather" in out
    assert "' rain'" in out and "top +logit" in out
    assert "[[" in out  # max token marked in activating text
    assert op.requests[0]["method"] == "GET"


def test_json_flag_prints_raw(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "--json", "feature", "gpt2-small", "0-res-jb", "14057")
    assert json.loads(out)["index"] == "14057"


def test_cache_avoids_second_request(env, capsys):
    op, make = env
    run(capsys, make(), "feature", "gpt2-small", "0-res-jb", "14057")
    run(capsys, make(), "feature", "gpt2-small", "0-res-jb", "14057")
    assert len(op.requests) == 1
    run(capsys, make(use_cache=False), "feature", "gpt2-small", "0-res-jb", "14057")
    assert len(op.requests) == 2


def test_search_bodies(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "search", "dogs")
    assert rc == 0 and "showing 3 of 20" in out and "--offset 20" in out
    assert op.requests[-1]["body"] == {"query": "dogs", "offset": 0}
    rc, out, _ = run(capsys, make(), "search", "dogs", "--model", "gpt2-small", "--source", "6-res-jb")
    assert op.requests[-1]["path"] == "/api/explanation/search"
    assert op.requests[-1]["body"]["layers"] == ["6-res-jb"]
    rc, _, err = run(capsys, make(), "search", "dogs", "--model", "gpt2-small")
    assert rc == 1 and "--source" in err


def test_topfeat_and_all(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "topfeat", "gpt2-small", "6-res-jb", "The cat", "-n", "50")
    assert rc == 0 and "1201" in out
    assert op.requests[-1]["body"]["numResults"] == 20  # clamped to the API maximum
    rc, out, _ = run(capsys, make(), "topfeat", "gpt2-small", "res-jb", "The cat sat", "--all", "-n", "2")
    assert rc == 0 and "11-res-jb" in out and "' cat'" in out
    assert op.requests[-1]["body"]["sourceSet"] == "res-jb"


def test_act(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "act", "gpt2-small", "6-res-jb", "100", "The cat sat on the mat")
    assert rc == 0 and "' mat'" in out
    assert op.requests[-1]["body"]["feature"]["index"] == "100"


def test_steer_body_and_not_cached(env, capsys):
    op, make = env
    argv = ["steer", "gpt2-small", "I think that", "--feature", "6-res-jb:100:20", "--tokens", "12"]
    rc, out, _ = run(capsys, make(), *argv)
    assert rc == 0 and "steered:" in out and "default:" in out
    b = op.requests[-1]["body"]
    assert b["features"] == [{"modelId": "gpt2-small", "layer": "6-res-jb", "index": 100, "strength": 20.0}]
    assert b["n_tokens"] == 12
    run(capsys, make(), *argv)
    assert len(op.requests) == 2  # steering is never cached
    rc, _, err = run(capsys, make(), "steer", "gpt2-small", "x", "--feature", "bad")
    assert rc == 1 and "SOURCE:INDEX:STRENGTH" in err


def test_lens(env, capsys):
    op, make = env
    rc, out, _ = run(capsys, make(), "lens", "qwen3.6-27b", "The capital of France is", "--layers", "0,54,63")
    assert rc == 0 and "JACOBIAN_LENS" in out and "L54" in out and "' Paris'" in out
    b = op.requests[-1]["body"]
    assert b["type"] == ["JACOBIAN_LENS"] and b["stream"] is False and b["prompt"]
    run(capsys, make(), "lens", "qwen3.6-27b", "hi", "--chat", "--type", "both", "--top", "99")
    b = op.requests[-1]["body"]
    assert b["chat"] == [{"role": "user", "content": "hi"}] and "prompt" not in b
    assert b["topN"] == 8 and b["type"] == ["LOGIT_LENS", "JACOBIAN_LENS"]


def test_pick_layers():
    assert npedia.pick_layers(list(range(64)), None) == [0, 9, 18, 27, 36, 45, 54, 63]
    assert npedia.pick_layers([0, 1, 2], None) == [0, 1, 2]
    assert npedia.pick_layers(list(range(10)), "3,99") == [3]


def test_headers_user_agent_and_key(env, capsys, monkeypatch):
    op, make = env
    run(capsys, make(use_cache=False), "act", "gpt2-small", "6-res-jb", "1", "x")
    h = {k.lower(): v for k, v in op.requests[-1]["headers"].items()}
    assert h["user-agent"] == "lab-npedia/0.1" and "x-api-key" not in h
    assert op.requests[-1]["timeout"] == 60.0
    monkeypatch.setenv(npedia.KEY_ENV, "SECRET-TEST-VALUE")
    rc, out, err = run(capsys, make(use_cache=False, base=npedia.BASE), "act", "gpt2-small", "6-res-jb", "1", "x")
    h = {k.lower(): v for k, v in op.requests[-1]["headers"].items()}
    assert h["x-api-key"] == "SECRET-TEST-VALUE"
    assert "SECRET-TEST-VALUE" not in out + err


def test_key_goes_only_to_neuronpedia(env, capsys, monkeypatch):
    op, make = env
    monkeypatch.setenv(npedia.KEY_ENV, "SECRET-TEST-VALUE")
    for base in ("https://neuronpedia.test", "https://www.neuronpedia.org.example.com", "http://127.0.0.1:9"):
        rc, out, err = run(capsys, make(use_cache=False, base=base), "act", "gpt2-small", "6-res-jb", "1", "x")
        h = {k.lower(): v for k, v in op.requests[-1]["headers"].items()}
        assert rc == 0 and "x-api-key" not in h, base
        assert "SECRET-TEST-VALUE" not in out + err


def test_redirect_to_another_host_drops_the_key():
    import urllib.request
    req = urllib.request.Request(npedia.BASE + "/api/feature/m/s/1", headers={"x-api-key": "SECRET-TEST-VALUE"})
    h = npedia._KeyStaysHome()
    away = h.redirect_request(req, None, 302, "Found", {}, "https://elsewhere.example/api/feature/m/s/1")
    assert away is not None and not [k for k in away.headers if k.lower() == "x-api-key"]
    home = h.redirect_request(req, None, 302, "Found", {}, "https://neuronpedia.org/api/feature/m/s/1")
    assert [k for k in home.headers if k.lower() == "x-api-key"]


def test_feature_index_zero_is_found(env, capsys):
    op, make = env
    op.routes["/api/feature/gpt2-small/0-res-jb/0"] = dict(SAMPLES["feature"], index=0)
    rc, out, _ = run(capsys, make(), "feature", "gpt2-small", "0-res-jb", "0")
    assert rc == 0 and "gpt2-small/0-res-jb/0 " in out
    op.routes["/api/feature/gpt2-small/0-res-jb/7"] = {"index": None}
    rc, _, err = run(capsys, make(), "feature", "gpt2-small", "0-res-jb", "7")
    assert rc == 1 and "feature not found" in err


def test_http_errors(env, capsys):
    op, make = env
    rc, _, err = run(capsys, make(), "feature", "gpt2-small", "0-res-jb", "1")
    assert rc == 1 and "HTTP 404" in err
    op.routes["/api/steer"] = urllib.error.HTTPError("u", 401, "unauth", {}, io.BytesIO(b"{}"))
    rc, _, err = run(capsys, make(), "steer", "m", "p", "--feature", "s:1:2")
    assert rc == 1 and npedia.KEY_ENV in err
    op.routes["/api/lens/prompt"] = {"error": "Validation error", "details": ["bad"]}
    rc, _, err = run(capsys, make(), "lens", "m", "p")
    assert rc == 1 and "Validation error" in err


def test_rate_limit_spacing(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(npedia.time, "sleep", lambda s: sleeps.append(s))
    npedia._rate_limit(str(tmp_path))
    npedia._rate_limit(str(tmp_path))
    assert sleeps and 0 < sleeps[-1] <= npedia.MIN_INTERVAL


def test_act_snippet_marks_max():
    s = npedia.act_snippet({"tokens": ["a", " b", " c"], "values": [0, 5, 1]})
    assert s == "a[[ b]] c"


def test_cache_root_uses_the_shared_folder_only_when_setup_marked_it(tmp_path, monkeypatch):
    for k in ("LAB_CACHE", "LAB_HOME"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    monkeypatch.setenv("HOME", str(tmp_path))
    default = str(tmp_path / ".cache" / "lab")
    assert npedia._cache_root() == default
    shared = tmp_path / "projects" / "lab-env"
    shared.mkdir(parents=True)
    assert npedia._cache_root() == default  # a folder that merely has the name
    (shared / ".lab-home").write_text("A lab home, made by `lab setup`.\n")
    assert npedia._cache_root() == str(shared / "cache")
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "one"))
    assert npedia._cache_root() == str(tmp_path / "one" / "cache")
    monkeypatch.setenv("LAB_CACHE", str(tmp_path / "two"))
    assert npedia._cache_root() == str(tmp_path / "two")
