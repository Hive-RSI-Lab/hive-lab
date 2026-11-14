# lab: small interpretability experiments, with baselines built in

`lab` is a workbench for agents. It runs quick experiments on small open-weight
language models: lenses, probes, steering, ablation and activation patching.
Each method reports its control next to the result. Use it to test an idea in
minutes on a CPU. It is not a tool for publishable claims.

Default model: `Qwen/Qwen3.5-0.8B` (24 layers, d_model 1024). Tests use `gpt2`.
Only ungated models are used. No HF token is needed.

## Quick start

```
lab setup            # venv, models, lenses, then check (see below for where they go)
lab check            # PASS/FAIL lines; exit 0 only if all pass
lab info             # which device "auto" picks and why
lab new myexp        # writes myexp.py (a probe sweep) here
lab run myexp.py     # one run at a time; stopped after 600 s (lab run --timeout 900 myexp.py for longer)
```

`lab setup` builds the venv with `uv` if the machine has it, else with `python3 -m venv` and pip
(Python 3.10 or newer; `LAB_NO_UV=1` forces that). It installs `requirements.txt` (torch is large:
allow about 6 GB of disk and a few minutes), then downloads the two models and their lenses
(about 2.5 GB) from Hugging Face. It needs `git` on the PATH, because `jlens` is installed from
its GitHub repository.

### Install

On a Hive: `hive module add lab` links the three commands (`lab`, `askbench`, `npedia`), then run
`lab setup` once.

Anywhere else, clone the repository and put the three scripts on your PATH under those names:

```
git clone https://github.com/Hive-RSI-Lab/hive-lab.git
ln -s "$PWD/hive-lab/lab.py" ~/.local/bin/lab
ln -s "$PWD/hive-lab/askbench.py" ~/.local/bin/askbench
ln -s "$PWD/hive-lab/npedia.py" ~/.local/bin/npedia
lab setup
```

`lab.py` and `npedia.py` need only Python's standard library; `lab setup` builds the venv that
the experiments run in. `askbench` and `npedia` have their own pages: `ASKBENCH.md`, `NPEDIA.md`.
Running on a rented GPU is described in `modal/README.md` (written from Modal's documentation and
not yet run against a live account).

### Where lab keeps its files

By default: the venv in `~/.local/share/lab/venv` (`LAB_VENV`), models in the Hugging Face cache
(`HF_HOME`), activations, fitted lenses and npedia lookups in `~/.cache/lab` (`LAB_CACHE`), settings
in `~/.config/lab/env` (`KEY=VALUE` lines; a variable that is already set wins).

`LAB_HOME=DIR` puts all of it in one folder: `DIR/venv`, `DIR/hf`, `DIR/cache` and `DIR/env`.
`lab setup` marks that folder with a file `.lab-home`.

**Where the programs that run lab are not given your environment** (agents in a sandbox with a private
home and a few shared folders, for example), do the setup once from an ordinary terminal, into a
folder they share:

```
LAB_HOME=~/projects/lab-env LAB_NO_UV=1 lab setup
```

When `LAB_HOME` is not set, lab looks for `~/projects/lab-env` and uses it if it holds the `.lab-home`
marker (a folder that merely has that name is left alone). After this one setup every such program
has the same venv, models and caches, and `lab check` passes for it. `LAB_NO_UV=1` builds the venv
with the system Python: a venv made by `uv` can point at a Python that `uv` keeps in your own home,
which the other programs may not see. Without a GPU there, lab uses the CPU.

### One run at a time

A run loads a model into memory, so two runs at once can hit the machine's memory limit. If the
machine has a job queue called `heavy` on the PATH (some Hives do), `lab run` and `lab check` go
through it. Without it, lab takes a file lock (`run.lock` in the lab cache): a second run prints a
line and waits for the first, and the wait does not count against `--timeout`. `lab py` takes no
lock. `LAB_NO_HEAVY=1` skips the queue and `LAB_NO_LOCK=1` skips the lock.

`lab py` is the venv's python with `lablib` importable, e.g. `lab py -m pytest tests/test_lablib.py`.
`lab models` lists known models. `lab fetch NAME` downloads one and its lens.

Results go to `./lab-runs/<UTC time>-<name>/`, or to `$LAB_RUN_DIR` if set.
Each holds `results.json`, `results.md`, a copy of the script, the seed,
the model name, the library versions and the command line.

A run folder is a working record, not something to publish as it is: it holds your script and
your prompts. In the saved command line, the value of an option named like a key, token, secret
or password is written as `***`; anything else you pass is saved as typed, so keep secrets in
the environment, not in arguments.

## Layer convention

Layer `L` is the OUTPUT of transformer block `L`, counting from 0. It is the
residual stream after `L+1` blocks. This is the jlens convention. In Hugging
Face terms it is `hidden_states[L+1]`. The exception is the last layer: HF
applies the final norm there, and lablib does not. There is no index for the raw
embeddings. Negative layers count from the end (`-1` = last block).

## Recipes

All snippets start with:

```python
import lablib
from lablib import acts, data, probe, steer, ablate, lens, patch, stats, report
m = lablib.load()            # or lablib.load("gpt2")
```

**Lens** (`examples/lens_compare.py`). It reads the residual through the logit lens and the Jacobian lens.
```python
res = lens.compare(m, "The Eiffel Tower is located in the city of", target=" Paris")
lens.print_compare(res)
```
The published neuronpedia lens is used when one exists. Otherwise a lens is
fitted on 16 prompts and labelled `SMALL-SAMPLE FIT`.

**Probe** (`examples/probe_sentiment.py`).
```python
ds = data.sentiment()
A = acts.get_acts(m, ds["prompts"], [6, 12, 18], position="last")
stats.print_table(probe.sweep_layers(A, ds["labels"], ds["groups"], control_task=True), probe.SWEEP_COLS)
```

**Steer** (`examples/steer_sentiment.py`).
```python
neutral_prompts = ["The weather over the weekend was", "The hotel room was"]   # no sentiment word yet
dm = probe.diff_means(A[12], ds["labels"], ds["groups"])
metric = steer.logit_diff_metric(m, " good", " bad")
r = steer.compare_to_random(m, neutral_prompts, 12, dm["raw"], alpha=1.0, metric=metric, k=20)
stats.print_table([r], steer.COMPARE_COLS)
```
Use `steer.steering(m, layer, vec, alpha)` as a context manager for anything else.

**Ablate** (`examples/ablate_sentiment.py`).
```python
r = ablate.compare_to_random(m, ds["prompts"], [12], dm["direction"], mode="mean",
                             metric=metric, ref_prompts=ds["prompts"], k=20)
```
`mode="project"` sets the component along the direction to zero. `mode="mean"`
holds it at its mean over `ref_prompts`. The default metric is KL divergence
from the clean run. `ablate.ablate_residual` zeroes the whole residual at given positions.

**Patch** (`examples/patch_capitals.py`).
```python
res = patch.patch_scan(m, "The capital of France is", "The capital of Italy is",
                       steer.logit_diff_metric(m, " Paris", " Rome"))
patch.print_scan(res)
```
The clean and corrupt prompts must have the same number of tokens.

**Your own data.** For prompt pairs, use
`data.pairs_from_template("The {x} cat sat", ["black"], ["white"])`.
Datasets are dicts with `prompts`, `labels` and `groups`.

## CPU times

These are wall times for the examples on CPU, measured with 8 threads while other jobs were using
the same machine, so an idle machine should be faster. They include model load but not time
spent waiting for another run.

| example | Qwen3.5-0.8B | gpt2 |
|---|---|---|
| lens_compare | 14 s | 22 s (cold start) |
| probe_sentiment | 48 s (first run; activations then cached) | 16 s |
| steer_sentiment (k=10) | 31 s | 10 s |
| ablate_sentiment (k=10) | 31 s | 11 s |
| patch_capitals | 23 s | 6 s |
| tests/test_lablib.py (gpt2) | | 7 s |

Generation is slower than scoring logits, because it runs one forward pass per new token.
The steering example's two 8-token generations fit inside its 31 s.

## Controls, and why they exist

- **Probe: majority class and shuffled labels.** A probe can fit noise in 1024
  dimensions, so the shuffled-label score shows what fitting noise looks like.
  Accuracy means little unless it is well above both.
- **Probe: grouped split.** Prompts made from one template or word stay on one side of the split.
  Without this, the probe can memorise near-duplicates. Always pass `groups`.
- **Probe: control task (selectivity).** After Hewitt & Liang (2019). With `control_task=True` two
  more probes are trained on one split in which every group appears on both sides: one on the
  real labels (`seen_acc`) and one on a made-up task where each group gets one random label
  (`control`). `selectivity = seen_acc - control`. Near 0 means the probe does as well on made-up
  labels as on real ones once it has seen each group, so its accuracy can come from recognising
  the groups (here, the words) and is no evidence of a feature. In the sentiment data the label
  is fixed by the adjective and every adjective is a group, so selectivity there is near 0 by
  design (gpt2: 0.00 at layers 0 to 6, 0.06 at layer 11). `acc`, the grouped held-out score, is
  the one that says whether the probe carries over to words it has not seen.
- **Steering: random vectors with the same norm.** Adding any large vector moves the logits.
  The comparison shows one thing: whether this direction moves the metric more than random
  directions of the same size do. It does not show that the direction is specific to the concept;
  for that, also measure a metric the direction should leave alone, and try a direction built
  from a different contrast.
- **Ablation: random directions.** Removing any direction changes outputs a little. It is the same
  test, with the same limit.
- **Lens: the logit lens next to the J-lens.** The last layer is the model's real output for both.
- **Patching: recovery is normalised.** The raw clean–corrupt gap is printed. When the gap is small, the ratio is noise.
- **Statistics:** bootstrap 95% CIs, permutation tests, and Holm correction for many layers (`stats.holm`).
  `p` against a random null cannot go below `1/(k+1)`. Read `z` as well, and raise `k` for a real test.

## What small-model results can and cannot support

They can support statements about this checkpoint, these prompts and this
layer. Examples: "a linear readout of X exists here", "adding this vector
shifts these logits more than random vectors do", "this site carries the
information in this prompt pair".

They cannot support statements about larger models, about the model "having
the concept" of X, or about mechanisms in general. Our datasets have tens to
hundreds of templated prompts, so they share surface features. Sentiment is
readable even at layer 0, because it is a property of the words. If a result
matters, repeat it with other templates, layers, seeds and models.

## Troubleshooting

- **Very slow forward passes.** Too many threads on a busy machine is the usual cause.
  lablib uses `min(8, cpus)` threads; set `LAB_THREADS=4` to go lower.
- **`lab run` waits a long time.** It waits for the run before it (see "One run at a time"). Do not skip the wait on a machine with a memory limit.
- **Out of memory.** Qwen3.5-0.8B needs about 3.8 GB of RAM in fp32. Use `lablib.load("gpt2")` on small machines.
- **Device.** `auto` picks CUDA only if free GPU memory is more than the bf16 model plus 1.5 GB.
  Set `LAB_DEVICE=cpu` or `cuda` to force it. `lab info` shows the decision.
- **Stale activations.** `get_acts` caches to `~/.cache/lab/acts` (`LAB_CACHE` moves it).
  Pass `cache=False`, or delete the folder.
- **Token ids.** `m.token_id(" Paris")` takes the FIRST token, with its leading space. Check with `m.tokens(text)`.
- **Qwen chat format.** `m.chat("question")` applies the chat template with thinking off.
- **Patching length error.** Pick words that are one token each; check with `m.tokens()`.

## Layout

`lab.py` (launcher, stdlib only), `requirements.txt`, `lablib/` (models, acts,
lens, probe, steer, ablate, patch, stats, data, report), `examples/`,
`askbench.py` with `askbench-examples/` (behavioural tests on a Claude model, see `ASKBENCH.md`),
`npedia.py` (Neuronpedia client, see `NPEDIA.md`), `modal/` (run a script on a Modal GPU),
`tests/`: `test_lablib.py` (gpt2; the `quick` subset runs in `lab check`), `test_launcher.py`,
`test_askbench.py` and `test_npedia.py` (both offline).

Run the tests with `lab py -m pytest tests`.
