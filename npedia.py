#!/usr/bin/env python3
"""npedia: small command-line client for Neuronpedia's public API (stdlib only).

Commands:  models, sources, feature, search, topfeat, act, steer, lens.  See NPEDIA.md.

- An API key is optional. It is read only from the environment variable
  NEURONPEDIA_API_KEY and sent as the x-api-key header to www.neuronpedia.org only (not to
  another host, and not along a redirect to one); it is never printed or saved.
- At most one request per second (shared across processes), 60 s timeout by default,
  User-Agent lab-npedia/0.1.
- Cache under ~/.cache/lab/npedia/ (LAB_CACHE moves it) for read-only lookups (resources page, features,
  searches). Steering and lens runs are not cached. --no-cache skips the cache.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://www.neuronpedia.org"
USER_AGENT = "lab-npedia/0.1"
KEY_ENV = "NEURONPEDIA_API_KEY"
KEY_HOSTS = ("www.neuronpedia.org", "neuronpedia.org")  # the only hosts the key is sent to


def _cache_root() -> str:
    """Where lab keeps its caches: $LAB_CACHE, else $LAB_HOME/cache, else ~/projects/lab-env/cache when
    `lab setup` made that folder (it holds the file .lab-home; processes that share it also share the rate
    limit), else ~/.cache/lab."""
    home = os.path.expanduser("~")
    if os.environ.get("LAB_CACHE"):
        return os.path.expanduser(os.environ["LAB_CACHE"])
    if os.environ.get("LAB_HOME"):
        return os.path.join(os.path.expanduser(os.environ["LAB_HOME"]), "cache")
    shared = os.path.join(home, "projects", "lab-env")
    marked = os.path.isfile(os.path.join(shared, ".lab-home"))
    return os.path.join(shared, "cache") if marked else os.path.join(home, ".cache", "lab")


CACHE_DIR = os.path.join(_cache_root(), "npedia")
MIN_INTERVAL = 1.0  # seconds between requests
DAY = 86400


class NpediaError(Exception):
    pass


# --------------------------------------------------------------------------- HTTP

def _rate_limit(cache_dir: str) -> None:
    """Sleep so that requests from any npedia process are >= MIN_INTERVAL apart."""
    os.makedirs(cache_dir, exist_ok=True)
    stamp = os.path.join(cache_dir, ".last_request")
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        fcntl = None
    with open(stamp, "a+") as fh:
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        try:
            last = float(fh.read().strip() or 0)
        except ValueError:
            last = 0.0
        wait = last + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        fh.seek(0)
        fh.truncate()
        fh.write(repr(time.time()))
        fh.flush()
        if fcntl:
            fcntl.flock(fh, fcntl.LOCK_UN)


class _KeyStaysHome(urllib.request.HTTPRedirectHandler):
    """Drop the API key when a reply redirects to a host that is not Neuronpedia."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname not in KEY_HOSTS:
            for k in [k for k in new.headers if k.lower() == "x-api-key"]:
                del new.headers[k]
        return new


class Client:
    def __init__(self, base: str = BASE, timeout: float = 60.0, cache_dir: str = CACHE_DIR,
                 use_cache: bool = True, opener=None):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.opener = opener or urllib.request.build_opener(_KeyStaysHome).open
        self.requests_made = 0

    # cache ------------------------------------------------------------------
    def _cache_path(self, method: str, path: str, body) -> str:
        key = json.dumps([method, self.base + path, body], sort_keys=True)
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return os.path.join(self.cache_dir, h + ".json")

    def _cache_get(self, p: str, ttl: float):
        if not self.use_cache or ttl <= 0:
            return None
        try:
            if time.time() - os.path.getmtime(p) > ttl:
                return None
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)["data"]
        except (OSError, ValueError, KeyError):
            return None

    def _cache_put(self, p: str, data, ttl: float) -> None:
        if not self.use_cache or ttl <= 0:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        tmp = p + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"data": data}, fh)
        os.replace(tmp, p)

    # request ----------------------------------------------------------------
    def request(self, method: str, path: str, body=None, ttl: float = 0, raw_text: bool = False):
        cp = self._cache_path(method, path, body)
        cached = self._cache_get(cp, ttl)
        if cached is not None:
            return cached
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        key = os.environ.get(KEY_ENV)
        if key and urllib.parse.urlsplit(self.base).hostname in KEY_HOSTS:  # never to another host
            headers["x-api-key"] = key
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        _rate_limit(self.cache_dir)
        self.requests_made += 1
        try:
            with self.opener(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            hint = ""
            if e.code in (401, 403) and not key:
                hint = f" (this endpoint may need an API key: set {KEY_ENV})"
            raise NpediaError(f"HTTP {e.code} from {method} {path}: {detail}{hint}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise NpediaError(f"request failed: {method} {path}: {e}") from None
        if raw_text:
            result = text
        else:
            try:
                result = json.loads(text)
            except ValueError:
                raise NpediaError(f"non-JSON reply from {path}: {text[:200]!r}") from None
            if isinstance(result, dict) and "error" in result and len(result) <= 3:
                raise NpediaError(f"{path}: {result.get('error')} {result.get('details', '')}")
        self._cache_put(cp, result, ttl)
        return result

    def get(self, path, ttl=0, raw_text=False):
        return self.request("GET", path, None, ttl, raw_text)

    def post(self, path, body, ttl=0):
        return self.request("POST", path, body, ttl)


# --------------------------------------------------------------------------- parsing

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")


def parse_resources(page: str) -> list[dict]:
    """Rows of the /available-resources table: model, source, inference (bool)."""
    out = []
    for row in _ROW.findall(page):
        cells = _CELL.findall(row)
        if len(cells) < 3:
            continue
        model = html.unescape(_TAG.sub("", cells[0])).strip()
        source = html.unescape(_TAG.sub("", cells[1])).strip()
        if not model or not source:
            continue
        out.append({"model": model, "source": source, "inference": "✅" in cells[2]})
    return out


def fetch_resources(c: Client) -> list[dict]:
    page = c.get("/available-resources", ttl=DAY, raw_text=True)
    rows = parse_resources(page)
    if not rows:
        raise NpediaError("could not parse /available-resources (page layout changed?)")
    return rows


# --------------------------------------------------------------------------- formatting

def _clip(s, n):
    s = "" if s is None else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def table(headers: list[str], rows: list[list], widths: list[int] | None = None) -> str:
    rows = [[("" if v is None else str(v)) for v in r] for r in rows]
    if widths:
        rows = [[_clip(v, w) if w else v for v, w in zip(r, widths)] for r in rows]
    cols = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(v.ljust(cols[i]) for i, v in enumerate(r)).rstrip()
    out = [line(headers), line(["-" * w for w in cols])]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _num(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "" if v is None else str(v)


def act_snippet(act: dict, width: int = 70) -> str:
    """The text of one activation record, with the max token marked [[like this]]."""
    toks = act.get("tokens") or []
    if not toks:
        return ""
    vals = act.get("values") or []
    mi = act.get("maxValueTokenIndex")
    if mi is None and vals:
        mi = max(range(len(vals)), key=lambda i: vals[i])
    mi = mi or 0
    # take a window around the max token
    left, right = mi, mi + 1
    length = len(toks[mi]) + 4
    while length < width and (left > 0 or right < len(toks)):
        if left > 0:
            left -= 1
            length += len(toks[left])
        if right < len(toks) and length < width:
            length += len(toks[right])
            right += 1
    parts = toks[left:mi] + ["[[" + toks[mi] + "]]"] + toks[mi + 1:right]
    s = "".join(parts).replace("\n", "\\n")
    return ("…" if left > 0 else "") + s + ("…" if right < len(toks) else "")


def fmt_models(rows: list[dict], filt: str | None) -> str:
    agg: dict[str, list[int]] = {}
    for r in rows:
        if filt and filt.lower() not in r["model"].lower():
            continue
        a = agg.setdefault(r["model"], [0, 0])
        a[0] += 1
        a[1] += r["inference"]
    body = [[m, n, k] for m, (n, k) in sorted(agg.items())]
    return table(["model", "sources", "inference"], body) + f"\n{len(body)} models"


def fmt_sources(rows: list[dict], model: str, filt: str | None) -> str:
    sel = [r for r in rows if r["model"] == model and (not filt or filt.lower() in r["source"].lower())]
    if not sel:
        raise NpediaError(f"no sources for model {model!r} (try: npedia models)")
    body = [[r["source"], "yes" if r["inference"] else "no"] for r in sel]
    return table(["source", "inference"], body) + f"\n{len(body)} sources"


def fmt_feature(f: dict, ntexts: int, nlogits: int) -> str:
    out = [f"{f.get('modelId')}/{f.get('layer')}/{f.get('index')}"
           f"  max act ~{_num(f.get('maxActApprox'))}"
           f"  density {_num((f.get('frac_nonzero') or 0) * 100, 4)}%"
           f"  {BASE}/{f.get('modelId')}/{f.get('layer')}/{f.get('index')}"]
    ex = f.get("explanations") or []
    out.append("")
    if ex:
        out.append(table(["explanation", "by"],
                         [[e.get("description"), e.get("explanationModelName") or e.get("typeName")] for e in ex],
                         [90, 24]))
    else:
        out.append("(no explanations)")
    pos = list(zip(f.get("pos_str") or [], f.get("pos_values") or []))[:nlogits]
    neg = list(zip(f.get("neg_str") or [], f.get("neg_values") or []))[:nlogits]
    if pos or neg:
        out.append("")
        n = max(len(pos), len(neg))
        rows = []
        for i in range(n):
            p = pos[i] if i < len(pos) else ("", None)
            q = neg[i] if i < len(neg) else ("", None)
            rows.append([repr(p[0]) if p[0] else "", _num(p[1]), repr(q[0]) if q[0] else "", _num(q[1])])
        out.append(table(["top +logit", "value", "top -logit", "value"], rows))
    acts = f.get("activations") or []
    if acts and ntexts:
        acts = sorted(acts, key=lambda a: -(a.get("maxValue") or 0))[:ntexts]
        out.append("")
        out.append(table(["max", "top activating text ([[max token]])"],
                         [[_num(a.get("maxValue")), act_snippet(a)] for a in acts]))
    return "\n".join(out)


def fmt_search(d: dict) -> str:
    res = d.get("results") or []
    rows = [[_num(r.get("cosine_similarity"), 3), r.get("modelId"), r.get("layer"), r.get("index"),
             r.get("description")] for r in res]
    s = table(["sim", "model", "source", "index", "explanation"], rows, [0, 24, 26, 8, 70])
    more = ""
    if d.get("hasMore"):
        more = f"  (more: --offset {d.get('nextOffset')})"
    total = d.get("resultsCount", len(res))
    return s + f"\nshowing {len(rows)} of {total} results{more}"


def _expl(feature: dict | None) -> str:
    ex = (feature or {}).get("explanations") or []
    return ex[0].get("description", "") if ex else ""


def fmt_topk(d: dict, per_token: int) -> str:
    rows = []
    for r in d.get("results") or []:
        for j, tf in enumerate((r.get("topFeatures") or [])[:per_token]):
            rows.append([r.get("position") if j == 0 else "", repr(r.get("token")) if j == 0 else "",
                         tf.get("featureIndex"), _num(tf.get("activationValue")), _expl(tf.get("feature"))])
    return f"source {d.get('source')}\n" + table(["pos", "token", "feature", "act", "explanation"], rows,
                                                [0, 16, 0, 0, 70])


def fmt_searchall(d: dict) -> str:
    rows = []
    toks = d.get("tokens") or []
    for r in d.get("result") or []:
        mi = r.get("maxValueIndex")
        tok = repr(toks[mi]) if isinstance(mi, int) and 0 <= mi < len(toks) else ""
        rows.append([r.get("layer"), r.get("index"), _num(r.get("maxValue")), tok, _expl(r.get("neuron"))])
    return table(["source", "index", "max act", "at token", "explanation"], rows, [26, 8, 0, 16, 60])


def fmt_act(d: dict) -> str:
    toks = d.get("tokens") or []
    vals = d.get("values") or []
    rows = [[i, repr(t), _num(v)] for i, (t, v) in enumerate(zip(toks, vals))]
    return (f"{d.get('modelId')}/{d.get('layer')}/{d.get('index')}  max {_num(d.get('maxValue'))}"
            f" at token {d.get('maxValueTokenIndex')}\n" + table(["pos", "token", "act"], rows))


def fmt_steer(d: dict, prompt: str) -> str:
    out = [f"prompt:   {prompt!r}",
           f"default:  {(d.get('DEFAULT') or '')!r}",
           f"steered:  {(d.get('STEERED') or '')!r}"]
    if d.get("shareUrl"):
        out.append(f"share:    {d['shareUrl']}")
        out.append("note:     Neuronpedia keeps this run, prompt included, under that public link; "
                   "put nothing private in a steer prompt.")
    if d.get("limit") is not None:
        out.append(f"(server-reported remaining steer limit: {d['limit']})")
    return "\n".join(out)


def pick_layers(all_layers: list[int], spec: str | None, n: int = 8) -> list[int]:
    if not all_layers:
        return []
    if spec:
        want = [int(x) for x in spec.split(",") if x.strip()]
        return [l for l in want if l in all_layers]
    if len(all_layers) <= n:
        return list(all_layers)
    step = (len(all_layers) - 1) / (n - 1)
    idx = sorted({round(i * step) for i in range(n)})
    return [all_layers[i] for i in idx]


def fmt_lens(d: dict, layers_spec: str | None) -> str:
    meta = d.get("meta") or {}
    toks = d.get("tokens") or []
    out = [f"model {meta.get('model')}  top_n {meta.get('top_n')}  "
           f"prompt tokens {meta.get('prompt_len')}  generated {meta.get('num_completion_tokens')}"]
    for typ, all_layers in (meta.get("layers_by_type") or {}).items():
        layers = pick_layers(all_layers, layers_spec)
        pos_in = [all_layers.index(l) for l in layers]
        rows = []
        for t in toks:
            res = next((r for r in t.get("results") or [] if r.get("type") == typ), None)
            if not res:
                continue
            tops = res.get("top_tokens") or []
            cells = [repr(tops[i][0]) if i < len(tops) and tops[i] else "" for i in pos_in]
            mark = "*" if t.get("is_generated") else ""
            rows.append([f"{t.get('position')}{mark}", repr(t.get("token"))] + cells)
        out.append("")
        out.append(f"{typ}: top-1 read-out per layer (rows = positions, * = generated)")
        out.append(table(["pos", "token"] + [f"L{l}" for l in layers], rows,
                         [0, 14] + [12] * len(layers)))
    return "\n".join(out)


# --------------------------------------------------------------------------- commands

def parse_feature_spec(s: str) -> tuple[str, int, float]:
    """SOURCE:INDEX:STRENGTH -> (source, index, strength)."""
    parts = s.rsplit(":", 2)
    if len(parts) != 3:
        raise NpediaError(f"--feature must be SOURCE:INDEX:STRENGTH, got {s!r}")
    try:
        return parts[0], int(parts[1]), float(parts[2])
    except ValueError:
        raise NpediaError(f"--feature must be SOURCE:INDEX:STRENGTH, got {s!r}") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="npedia", description="Neuronpedia API client (see NPEDIA.md).")
    p.add_argument("--json", action="store_true", help="print the raw JSON response")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    p.add_argument("--timeout", type=float, default=60.0, help="seconds per request (default 60)")
    p.add_argument("--base", default=BASE, metavar="URL",
                   help="another Neuronpedia server, e.g. one you host (default %(default)s); the API key is "
                        "never sent to it")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("models", help="list models (from /available-resources)")
    s.add_argument("filter", nargs="?", help="substring filter")

    s = sub.add_parser("sources", help="list sources (SAEs/transcoders) of one model")
    s.add_argument("model")
    s.add_argument("filter", nargs="?", help="substring filter")

    s = sub.add_parser("feature", help="one feature: explanations, top logits, top activating texts")
    s.add_argument("model")
    s.add_argument("source", help="e.g. 6-res-jb")
    s.add_argument("index", type=int)
    s.add_argument("--texts", type=int, default=5, help="top activating texts to show (default 5)")
    s.add_argument("--logits", type=int, default=8, help="top logits to show (default 8)")

    s = sub.add_parser("search", help="search feature explanations by text")
    s.add_argument("query")
    s.add_argument("--model", help="restrict to one model (needs --source)")
    s.add_argument("--source", action="append", help="source(s) within --model; repeatable")
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("-n", type=int, default=20, help="rows to show (default 20)")

    s = sub.add_parser("topfeat", help="top features for a custom text")
    s.add_argument("model")
    s.add_argument("source", help="a source (per-token top features), or a source set with --all")
    s.add_argument("text")
    s.add_argument("-n", type=int, default=3, help="features per token (max 20; default 3)")
    s.add_argument("--all", action="store_true",
                   help="treat SOURCE as a source set (e.g. res-jb) and search all its layers; "
                        "shows the top -n features overall (default 3 -> use -n 10)")

    s = sub.add_parser("act", help="activations of one feature on a custom text")
    s.add_argument("model")
    s.add_argument("source")
    s.add_argument("index", type=int)
    s.add_argument("text")

    s = sub.add_parser("steer", help="generate with and without steering")
    s.add_argument("model")
    s.add_argument("prompt")
    s.add_argument("--feature", action="append", required=True, metavar="SOURCE:INDEX:STRENGTH",
                   help="feature to add; repeatable")
    s.add_argument("--tokens", type=int, default=32, help="tokens to generate (default 32)")
    s.add_argument("--temperature", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=16)
    s.add_argument("--multiplier", type=float, default=4.0, help="strength_multiplier (default 4)")
    s.add_argument("--method", default="SIMPLE_ADDITIVE", choices=["SIMPLE_ADDITIVE", "ORTHOGONAL_DECOMP"])

    s = sub.add_parser("lens", help="Jacobian / logit lens over a prompt")
    s.add_argument("model", help="e.g. qwen3.6-27b")
    s.add_argument("prompt")
    s.add_argument("--type", default="jacobian", choices=["jacobian", "logit", "both"])
    s.add_argument("--top", type=int, default=3, help="top tokens per layer requested (1-8, default 3)")
    s.add_argument("--gen", type=int, default=0, help="tokens to generate after the prompt (default 0)")
    s.add_argument("--layers", help="comma list of layers to show (default: 8 evenly spaced)")
    s.add_argument("--chat", action="store_true", help="send PROMPT as one user turn in chat format")
    return p


def run(args, c: Client) -> tuple[object, str]:
    """Returns (raw response, formatted text)."""
    if args.cmd in ("models", "sources"):
        rows = fetch_resources(c)
        if args.cmd == "models":
            return rows, fmt_models(rows, args.filter)
        return [r for r in rows if r["model"] == args.model], fmt_sources(rows, args.model, args.filter)

    if args.cmd == "feature":
        path = "/api/feature/{}/{}/{}".format(*(urllib.parse.quote(str(x), safe="")
                                                for x in (args.model, args.source, args.index)))
        d = c.get(path, ttl=7 * DAY)
        if not isinstance(d, dict) or d.get("index") in (None, ""):  # index 0 is a real feature
            raise NpediaError(f"feature not found: {args.model}/{args.source}/{args.index}")
        return d, fmt_feature(d, args.texts, args.logits)

    if args.cmd == "search":
        if args.model:
            if not args.source:
                raise NpediaError("--model needs at least one --source")
            body = {"modelId": args.model, "layers": args.source, "query": args.query, "offset": args.offset}
            d = c.post("/api/explanation/search", body, ttl=DAY)
        else:
            d = c.post("/api/explanation/search-all", {"query": args.query, "offset": args.offset}, ttl=DAY)
        shown = dict(d, results=(d.get("results") or [])[: args.n],
                     resultsCount=d.get("resultsCount", len(d.get("results") or [])))
        return d, fmt_search(shown)

    if args.cmd == "topfeat":
        if args.all:
            body = {"modelId": args.model, "sourceSet": args.source, "text": args.text,
                    "selectedLayers": [], "sortIndexes": [], "ignoreBos": True,
                    "densityThreshold": -1, "numResults": max(1, args.n)}
            d = c.post("/api/search-all", body, ttl=DAY)
            return d, fmt_searchall(d)
        n = min(max(1, args.n), 20)
        body = {"modelId": args.model, "source": args.source, "text": args.text, "numResults": n}
        d = c.post("/api/search-topk-by-token", body, ttl=DAY)
        return d, fmt_topk(d, n)

    if args.cmd == "act":
        body = {"feature": {"modelId": args.model, "source": args.source, "index": str(args.index)},
                "customText": args.text}
        d = c.post("/api/activation/new", body, ttl=DAY)
        return d, fmt_act(d)

    if args.cmd == "steer":
        feats = []
        for spec in args.feature:
            src, idx, strength = parse_feature_spec(spec)
            feats.append({"modelId": args.model, "layer": src, "index": idx, "strength": strength})
        body = {"prompt": args.prompt, "modelId": args.model, "features": feats,
                "temperature": args.temperature, "n_tokens": args.tokens, "seed": args.seed,
                "strength_multiplier": args.multiplier, "steer_method": args.method}
        d = c.post("/api/steer", body)
        return d, fmt_steer(d, args.prompt)

    if args.cmd == "lens":
        types = {"jacobian": ["JACOBIAN_LENS"], "logit": ["LOGIT_LENS"],
                 "both": ["LOGIT_LENS", "JACOBIAN_LENS"]}[args.type]
        body = {"modelId": args.model, "type": types, "topN": min(max(1, args.top), 8),
                "numCompletionTokens": max(0, args.gen), "stream": False}
        if args.chat:
            body["chat"] = [{"role": "user", "content": args.prompt}]
        else:
            body["prompt"] = args.prompt
        d = c.post("/api/lens/prompt", body)
        return d, fmt_lens(d, args.layers)

    raise NpediaError(f"unknown command {args.cmd}")  # pragma: no cover


def main(argv=None, client: Client | None = None) -> int:
    args = build_parser().parse_args(argv)
    c = client or Client(base=args.base, timeout=args.timeout, use_cache=not args.no_cache)
    try:
        raw, text = run(args, c)
    except NpediaError as e:
        print(f"npedia: {e}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(raw, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
