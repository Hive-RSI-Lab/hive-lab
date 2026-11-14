"""lablib: small interpretability experiments on small open-weight models.

Layer convention: layer L = output of transformer block L (0-based), the
residual stream after L+1 blocks, as in the jlens library. See README.md.
"""

import os as _os

# Too many threads on a busy machine is very slow (measured: one Qwen3.5-0.8B
# forward took 50 s with 20 threads on a loaded 20-core machine, 0.4 s with 8).
# LAB_THREADS overrides; default min(8, cpus).
_n = int(_os.environ.get("LAB_THREADS", min(8, _os.cpu_count() or 1)))
try:
    import torch as _torch

    _torch.set_num_threads(_n)
except ImportError:  # launcher-only use without the venv
    pass

from .models import DEFAULT, Model, load  # noqa: F401

__version__ = "0.1.0"
