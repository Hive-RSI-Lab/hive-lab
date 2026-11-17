"""Says when Modal's client would fail behind a proxy (hive-lab #1).

Behind an HTTP CONNECT proxy (as inside a Hive's sandbox, where every connection goes
through the proxy named in HTTPS_PROXY), the modal client reaches its server only when
`python-socks` and `aiohttp-socks` are installed: they come with
`pip install 'modal[api-proxy-support]'`, not with plain `pip install modal`. Without them
every command fails with "Could not connect to the Modal server", which hides the cause.

The rule below follows modal 1.6's own (modal/_utils/proxy_support.py): the proxy comes from
urllib's `<scheme>_proxy` variables (lowercase wins), and since Modal is reached over TLS only
`https_proxy` or `all_proxy` count, not `http_proxy`. The gRPC channel to the API needs
python-socks unless NO_PROXY covers the API host; the HTTPS session for file transfers needs
aiohttp-socks whenever a proxy is set (it does not consult NO_PROXY). MODAL_DISABLE_API_PROXY
turns both off. No modal import here, so the test runs without the client.
"""
import importlib.util
import os
import urllib.request

API_HOST = "api.modal.com"
PIP_NAME = {"python_socks": "python-socks", "aiohttp_socks": "aiohttp-socks"}


def _proxies(env):
    """urllib.request.getproxies_environment(), over a given mapping instead of os.environ."""
    proxies, lower = {}, []
    for name, value in env.items():
        if len(name) > 5 and name[-6] == "_" and name[-5:].lower() == "proxy":
            if value:
                proxies[name[:-6].lower()] = value
            if name[-6:] == "_proxy":
                lower.append((name[:-6], value))
    if "REQUEST_METHOD" in env:
        proxies.pop("http", None)
    for key, value in lower:  # a lowercase variable wins, even when empty
        if value:
            proxies[key] = value
        else:
            proxies.pop(key, None)
    return proxies


def proxy_warning(env=None, have=None):
    """Returns a warning line when modal would use a proxy and a package it needs is missing, else None."""
    env = os.environ if env is None else env
    have = (lambda m: importlib.util.find_spec(m) is not None) if have is None else have
    if str(env.get("MODAL_DISABLE_API_PROXY", "")).lower() not in {"", "0", "false"}:
        return None
    proxies = _proxies(env)
    key = "https" if proxies.get("https") else "all" if proxies.get("all") else None
    if not key:
        return None
    needed = ["aiohttp_socks"]
    if not urllib.request.proxy_bypass_environment(API_HOST, proxies):
        needed.insert(0, "python_socks")
    missing = [PIP_NAME[m] for m in needed if not have(m)]
    if not missing:
        return None
    return (f"proxy: {key}_proxy is set, but {' and '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not installed, which modal needs to go through "
            "it (the visible error is \"Could not connect to the Modal server\"). "
            "Install with: pip install 'modal[api-proxy-support]'")
