Untested against a live account.

# Running lab experiments on a Modal GPU

`run_remote.py` runs one local experiment script on a [Modal](https://modal.com) GPU and copies
its run folder back. Everything below was written from Modal's public pages and checked offline
(syntax, and `python modal/run_remote.py --check` with modal 1.6.4 and no token) on 2026-11-14.
Nothing has been run on Modal itself.

## What the app does

- Image: Debian slim with Python 3.12; installs the module's `requirements.txt` if it exists;
  adds `lablib/` under `/root/lab` with `PYTHONPATH=/root/lab`, so `import lablib` works.
- One function, `run_experiment`: GPU T4 by default (`--gpu L4`, `A10`, `A100`, `A100-40GB`,
  `A100-80GB`), timeout 20 minutes by default (`--timeout-min`), and a Modal Volume
  `lab-hf-cache` mounted as the Hugging Face cache (`HF_HOME=/cache/hf`), so model downloads are
  kept between runs (Volume storage is billed, see prices).
- The script's file is sent to the container and run as `python SCRIPT ARGS` with `LAB_RUN_DIR`
  set to an empty folder. The script must write its results there. When it ends, the folder comes
  back as a `.tar.gz` and is unpacked into `--out` (default `runs/<script>-remote-<time>`).
  Only regular files inside that folder are unpacked.
- Only the script file itself and `lablib/` are sent: if the script reads other local files,
  they are not there.

## Account steps

1. Sign up at <https://modal.com/signup> (offers GitHub, Google or SSO sign-in).
2. Install the client locally: `pip install modal` (into a venv).
3. Sign in: `modal setup` (opens a browser), or `modal token new`. Both write a token file,
   `~/.modal.toml`. **That file is a credential: never print it, copy it, paste it into a chat
   or commit it.** The same applies to `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` if you use them.
4. Set a spending limit before the first GPU run (see below).

## How to run

```sh
cd lab                                                  # the repo root
python modal/run_remote.py --check                      # offline: no token, no network
modal run modal/run_remote.py --script examples/my_exp.py
modal run modal/run_remote.py --script examples/my_exp.py --gpu L4 --timeout-min 40 \
    --args "--n 200" --out runs/my_exp-l4
```

The first run builds the image (installs torch etc.), which takes several minutes and is billed
as CPU time. `modal run` starts a temporary app that stops when the command ends.

## Prices (from <https://modal.com/pricing>, read on 2026-11-14)

Modal bills per second. Per hour = per-second price × 3600.

| GPU | per second | per hour |
|---|---|---|
| T4 | $0.000164 | $0.59 |
| L4 | $0.000222 | $0.80 |
| A10 | $0.000306 | $1.10 |
| A100, 40 GB | $0.000583 | $2.10 |
| A100, 80 GB | $0.000694 | $2.50 |

On top of the GPU: CPU $0.0000131 per physical core per second (at least 0.125 core per
container), memory $0.00000222 per GiB per second, Volumes $0.09 per GiB per month (the page
says 1 TiB per month is included). Choosing a region costs 1.15–1.75× base prices. The page
lists plain `A100` only as the 40 GB and 80 GB rows; use the explicit name if the price matters.

Plans on that date: **Starter** $0 per month plus compute, with $30 of free compute per month
(up to 10 GPUs at once); Team $250 per month with $100 credits. The pricing FAQ says usage is
pay-as-you-go and no credits must be bought in advance.

**Prices, free credits and plans can change at any time; check the pricing page before relying
on these numbers.**

## Payment card

- Sign-up: whether a payment card is needed to sign up is **not confirmed**. The sign-up page
  shows only GitHub, Google and SSO buttons and does not mention a card.
- GPUs: Modal's GPU guide says "Using a GPU requires having a valid payment method on file."
  So this app (which always uses a GPU) will need a card or other payment method on the account,
  even if usage stays inside the free $30.

## Spending limit

From <https://modal.com/docs/guide/budgets> (2026-11-14): on the **Usage & Billing** page a
Workspace Owner or Manager can set

- a **Workspace budget**: a monthly cap on total usage (before credits), and
- a **spend limit**: a monthly cap on out-of-pocket charges after credits. When it is reached,
  Modal stops workloads that would cost more. If not set, it defaults to the usage limit minus
  credits.

Set both low (for example a few dollars) before the first run. The timeout (default 20 min) also
bounds each run: a T4 run that hits it costs about $0.20 of GPU time plus CPU and memory.

## How to stop everything

- Ctrl-C in the terminal running `modal run` stops the temporary app and its containers.
- `modal app list` shows running apps; `modal app stop APP_ID` stops one (also the red
  "Stop app" button on the app's page in the dashboard).
- `modal container list` / `modal container stop CONTAINER_ID` for single containers.
- To stop paying for storage: `modal volume delete lab-hf-cache`.
- Last resort: lower the spend limit to $0 on the Usage & Billing page.

## What was and was not checked

Checked offline on 2026-11-14 (modal 1.6.4 in a scratch venv, empty HOME, no token): the file
compiles; `--check` builds the app, the image definition (requirements found, lablib found),
the GPU variants via `with_options`, and the copy-back unpacking (a `../` path is skipped);
`modal run modal/run_remote.py --help` lists the options. Not checked: image build, GPU run,
Volume, copy-back over the network, real costs.
