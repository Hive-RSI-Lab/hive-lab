"""Says when Modal's client would fail behind a proxy (hive-lab #1).

Behind an HTTP CONNECT proxy (as inside a Hive's sandbox, where every connection goes
through the proxy named in HTTPS_PROXY), the modal client reaches its server only when
`python-socks` and `aiohttp-socks` are installed: they come with
`pip install 'modal[api-proxy-support]'`, not with plain `pip install modal`. Without them
every command fails with "Could not connect to the Modal server", which hides the cause.
No modal import here, so the test runs without the client.
"""
import importlib.util
import os

PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy")
NEEDED = ("python_socks", "aiohttp_socks")


def proxy_warning(env=None, have=None):
    """Returns a warning line when a proxy is set and a package it needs is missing, else None."""
    env = os.environ if env is None else env
    have = (lambda m: importlib.util.find_spec(m) is not None) if have is None else have
    var = next((v for v in PROXY_VARS if env.get(v)), None)
    if not var:
        return None
    missing = [m for m in NEEDED if not have(m)]
    if not missing:
        return None
    return (f"proxy: {var} is set but {', '.join(missing)} is not installed, so every modal "
            "command will fail with \"Could not connect to the Modal server\". "
            "Install with: pip install 'modal[api-proxy-support]'")
