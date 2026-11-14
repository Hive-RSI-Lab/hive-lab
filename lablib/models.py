"""Load small open-weight language models for experiments.

`load(name)` returns a `Model`: the Hugging Face model, its tokenizer, the
jlens wrapper (`.lm`), and a few helpers.

Layer convention used everywhere in lablib (same as the jlens library):
layer L means the OUTPUT of transformer block L (0-based), i.e. the residual
stream after L+1 blocks. Layer n_layers-1 is the last block's output, BEFORE
the final norm. There is no layer index for the raw embeddings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass

DEFAULT = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "main"

# params = text-model parameters (what gets loaded for causal LM use).
KNOWN = {
    "openai-community/gpt2": dict(
        params=124e6, lens="gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
        note="tiny, for tests"),
    "EleutherAI/pythia-70m-deduped": dict(
        params=70e6,
        lens="pythia-70m-deduped/jlens/Salesforce-wikitext/pythia-70m-deduped_jacobian_lens.pt",
        note="tiny; less tested"),
    "Qwen/Qwen3.5-0.8B": dict(
        params=752e6, lens="qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
        note="default; chat model"),
    "Qwen/Qwen3.5-2B": dict(
        params=1.9e9, lens="qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt",
        note="optional; slow on CPU"),
}
ALIASES = {
    "gpt2": "openai-community/gpt2",
    "pythia-70m": "EleutherAI/pythia-70m-deduped",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
}
GPU_MARGIN_BYTES = 1.5e9


def canonical(name: str) -> str:
    return ALIASES.get(name, ALIASES.get(name.lower(), name))


def n_params(name: str) -> float | None:
    name = canonical(name)
    if name in KNOWN:
        return KNOWN[name]["params"]
    try:
        from huggingface_hub import get_safetensors_metadata

        return float(sum(get_safetensors_metadata(name).parameter_count.values()))
    except Exception:
        return None


def ram_needed_bytes(name: str, bytes_per_param: int = 4) -> float | None:
    """Rough peak memory: weights x 1.25 (activations, logits, tokenizer)."""
    p = n_params(name)
    return None if p is None else p * bytes_per_param * 1.25


def gpu_free_bytes() -> tuple[int | None, str]:
    """Free memory on GPU 0, read with nvidia-smi so we never open a CUDA
    context just to ask. Returns (bytes or None, how we know)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None, "no nvidia-smi on PATH"
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0]) * 1024 * 1024, "nvidia-smi"
    except Exception as e:  # noqa: BLE001
        return None, f"nvidia-smi failed: {e}"


def pick_device(name: str = DEFAULT, device: str = "auto") -> tuple[str, str]:
    """Return (device, reason). "auto" picks CUDA only if free GPU memory
    exceeds the model's bf16 needs plus a 1.5 GB margin. LAB_DEVICE overrides."""
    env = os.environ.get("LAB_DEVICE")
    if device == "auto" and env:
        return env, f"LAB_DEVICE={env}"
    if device != "auto":
        return device, "requested explicitly"
    free, how = gpu_free_bytes()
    if free is None:
        return "cpu", f"cpu: no GPU info ({how})"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu", "cpu: torch sees no CUDA device"
    except Exception:
        return "cpu", "cpu: torch import failed"
    need = ram_needed_bytes(name, 2)
    if need is None:
        return "cpu", "cpu: unknown model size"
    if free > need + GPU_MARGIN_BYTES:
        return "cuda", f"cuda: {free/1e9:.1f} GB free > {need/1e9:.1f} GB needed + 1.5 GB margin"
    return "cpu", f"cpu: GPU has {free/1e9:.1f} GB free, needs {need/1e9:.1f} GB + 1.5 GB margin"


@dataclass
class Model:
    """A loaded model. Attributes: hf (HF model), tok (tokenizer), lm (jlens
    HFLensModel), name, n_layers, d_model, device, dtype, layers (block list),
    device_reason."""

    hf: object
    tok: object
    lm: object
    name: str
    n_layers: int
    d_model: int
    device: str
    dtype: object
    device_reason: str

    @property
    def layers(self):
        return self.lm.layers

    def __repr__(self):
        return (f"Model({self.name}, n_layers={self.n_layers}, d_model={self.d_model}, "
                f"device={self.device}, dtype={self.dtype})")

    # --- tokenising -------------------------------------------------------
    def encode(self, texts, max_length: int = 512):
        """Tokenise a list of strings with RIGHT padding. Returns a dict with
        input_ids, attention_mask (on device) and lengths (list of ints).
        Right padding keeps every real token's activations identical to an
        unbatched run (the models are causal)."""
        if isinstance(texts, str):
            texts = [texts]
        self.tok.padding_side = "right"
        enc = self.tok(list(texts), return_tensors="pt", padding=True,
                       truncation=True, max_length=max_length)
        lengths = enc["attention_mask"].sum(1).tolist()
        return {"input_ids": enc["input_ids"].to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device),
                "lengths": [int(x) for x in lengths]}

    def tokens(self, text: str) -> list[str]:
        ids = self.encode([text])["input_ids"][0]
        return [self.tok.decode([int(i)]) for i in ids]

    def token_id(self, s: str) -> int:
        """Id of the first token of `s` (no BOS). Use " Paris" (with the
        leading space) for a word that follows other text."""
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"{s!r} tokenises to nothing")
        return int(ids[0])

    def chat(self, user: str, system: str | None = None, assistant_prefill: str = "",
             thinking: bool = False) -> str:
        """Format one user turn with the chat template (if the model has one).
        thinking=False asks Qwen3.5-style templates to skip the think block."""
        if not getattr(self.tok, "chat_template", None):
            return (system + "\n\n" if system else "") + user + "\n" + assistant_prefill
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        kw = {"enable_thinking": thinking}
        if assistant_prefill:
            msgs.append({"role": "assistant", "content": assistant_prefill})
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                continue_final_message=True, **kw)
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, **kw)

    # --- running ----------------------------------------------------------
    def last_logits(self, texts, batch_size: int = 8):
        """Next-token logits at the last real token of each text: [n, vocab] float32 CPU."""
        import torch

        if isinstance(texts, str):
            texts = [texts]
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.encode(texts[i:i + batch_size])
                logits = self.hf(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"], use_cache=False).logits
                idx = torch.tensor(enc["lengths"], device=logits.device) - 1
                outs.append(logits[torch.arange(len(idx)), idx].float().cpu())
        return torch.cat(outs)

    def generate(self, prompt: str, max_new_tokens: int = 30, greedy: bool = True,
                 temperature: float = 1.0, seed: int = 0) -> str:
        """Generate a continuation for one prompt; returns only the new text."""
        import torch

        torch.manual_seed(seed)
        enc = self.encode([prompt])
        with torch.no_grad():
            out = self.hf.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=not greedy,
                temperature=temperature if not greedy else None,
                top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def load(name: str = DEFAULT, device: str = "auto", dtype: str = "auto",
         verbose: bool = True) -> Model:
    """Load a model for text-only causal-LM use.

    device: "auto" | "cpu" | "cuda". dtype: "auto" (float32 on CPU, bfloat16
    on CUDA) | "float32" | "bfloat16" | "float16".
    Qwen3.5 checkpoints are vision-language; AutoModelForCausalLM loads only
    the text decoder (Qwen3_5ForCausalLM), as the jlens walkthrough does.
    """
    import torch
    import transformers

    transformers.logging.set_verbosity_error()
    try:
        transformers.utils.logging.disable_progress_bar()
    except Exception:
        pass
    import jlens

    name = canonical(name)
    dev, reason = pick_device(name, device)
    if dtype == "auto":
        tdtype = torch.float32 if dev == "cpu" else torch.bfloat16
    else:
        tdtype = getattr(torch, dtype)
    if verbose:
        print(f"[lab] loading {name} on {dev} ({reason}), dtype {str(tdtype).replace('torch.', '')}",
              flush=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=tdtype)
    hf.to(dev)
    tok = transformers.AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = jlens.from_hf(hf, tok)  # sets eval(), requires_grad False, BOS like the lens fit
    return Model(hf=hf, tok=tok, lm=lm, name=name, n_layers=lm.n_layers,
                 d_model=lm.d_model, device=dev, dtype=tdtype, device_reason=reason)


@contextmanager
def block_hook(model: Model, layer: int, fn):
    """Register fn(hidden) -> hidden|None as a forward hook on block `layer`.
    `hidden` is the block's output residual [batch, seq, d_model]."""

    def hook(module, inputs, output):
        h = output if not isinstance(output, tuple) else output[0]
        new = fn(h)
        if new is None:
            return None
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    handle = model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def fetch(name: str = DEFAULT, lens: bool = True) -> None:
    """Download a model (and its published lens, if any) into the HF cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    name = canonical(name)
    print(f"[lab] fetching {name} ...", flush=True)
    snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                            "*.jinja", "*.tiktoken"])
    if lens and name in KNOWN and KNOWN[name].get("lens"):
        print(f"[lab] fetching lens {KNOWN[name]['lens']} ...", flush=True)
        hf_hub_download(LENS_REPO, KNOWN[name]["lens"], revision=LENS_REVISION)
    print(f"[lab] {name}: done", flush=True)


def is_cached(name: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        name = canonical(name)
        return any(r.repo_id == name and r.repo_type == "model" for r in scan_cache_dir().repos)
    except Exception:
        return False


def lens_cached(name: str) -> bool:
    name = canonical(name)
    if name not in KNOWN:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(LENS_REPO, KNOWN[name]["lens"], revision=LENS_REVISION)
        return isinstance(p, str)
    except Exception:
        return False
