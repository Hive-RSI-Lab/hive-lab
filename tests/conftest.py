import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "quick: fast subset run by `lab check`")


@pytest.fixture(autouse=True, scope="session")
def lab_cache_in_tmp(tmp_path_factory):
    """Point LAB_CACHE at a temporary folder, so no test writes into the real lab cache."""
    mp = pytest.MonkeyPatch()
    mp.setenv("LAB_CACHE", str(tmp_path_factory.mktemp("lab-cache")))
    yield
    mp.undo()
