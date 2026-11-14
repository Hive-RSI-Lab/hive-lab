# npedia: Neuronpedia from the command line

`npedia` is a small client for [Neuronpedia](https://www.neuronpedia.org)'s public API
(stdlib Python only). It lists models and sources, looks up features, searches explanations,
finds the top features for your own text, steers, and runs the Jacobian / logit lens.

## Quick start

```sh
npedia models gemma                                   # models whose id contains "gemma"
npedia sources gpt2-small res-jb                      # sources (SAEs) of one model
npedia feature gpt2-small 6-res-jb 3986               # one feature
npedia search "references to dogs"                    # search explanations
npedia topfeat gpt2-small 6-res-jb "The cat sat on the mat"
npedia steer gpt2-small "I think that" --feature 6-res-jb:3986:20
npedia lens qwen3.6-27b "The capital of France is" --gen 2
```

Global options (before the command): `--json` prints the raw API response instead of a table;
`--no-cache` skips the cache; `--timeout SEC` (default 60); `--base URL` talks to another
Neuronpedia server, such as one you host (the API key is never sent to it).

Behaviour: at most one request per second (shared by all npedia processes), User-Agent
`lab-npedia/0.1`. Read-only lookups are cached in the lab cache, folder `npedia/` (`~/.cache/lab/npedia/` by
default; `LAB_CACHE` or `LAB_HOME` moves it): the resource list, searches, `topfeat` and `act` results for a day,
features for a week. Steering and lens runs are never cached.

## API key

No command needs a key: on 2026-11-14 every command below worked with no key, including the
endpoints the API docs mark as needing one. If you have a key, put it in the environment
variable `NEURONPEDIA_API_KEY`; npedia sends it as the `x-api-key` header, to neuronpedia.org and to
no other host. It is read only from the environment, never from a file, and never printed or cached. Neuronpedia may start requiring
a key at any time; npedia then says so in the error message.

| command | endpoint | API docs say | worked without a key (2026-11-14) |
|---|---|---|---|
| `models`, `sources` | page `/available-resources` (parsed; there is no list endpoint) | (web page) | yes |
| `feature` | `GET /api/feature/{model}/{source}/{index}` | key required | yes |
| `search` | `POST /api/explanation/search-all`, or `/api/explanation/search` with `--model` | key required | yes |
| `topfeat` | `POST /api/search-topk-by-token`; with `--all`: `POST /api/search-all` | none; `--all`: key required | yes |
| `act` | `POST /api/activation/new` | none | yes |
| `steer` | `POST /api/steer` | key optional | yes (reply includes a server "limit" counter: 239 then 238) |
| `lens` | `POST /api/lens/prompt` (`stream: false`) | none | yes |

Only models whose sources have "inference" enabled (see `npedia sources`) can be used with
`topfeat`, `act` and `steer`. The lens works with the models the J-lens page offers
(`qwen3.6-27b` was tested).

## Commands, with example output

The outputs below show the layout of each table. The model and source ids are real; the numbers,
explanations and texts are made-up examples, so run the command to see what Neuronpedia returns.

### models / sources

```
$ npedia models gemma
model            sources  inference
---------------  -------  ---------
gemma-2-2b       300      140
gemma-2-2b-it    27       27
gemma-2-9b       380      0
...
```

`inference` is the number of sources with inference enabled.

```
$ npedia sources gpt2-small res-jb
source     inference
---------  ---------
0-res-jb   yes
1-res-jb   yes
...
```

### feature

Explanations, top positive/negative logits, and top activating texts with the strongest token
in `[[ ]]`. `--texts N` and `--logits N` set how many rows to show.

```
$ npedia feature gpt2-small 6-res-jb 3986 --texts 3 --logits 5
gpt2-small/6-res-jb/3986  max act ~45.50  density 0.0350%  https://www.neuronpedia.org/gpt2-small/6-res-jb/3986

explanation                    by
-----------------------------  -----------
words about dogs               explainer-a
dogs and other household pets  explainer-b

top +logit  value  top -logit  value
----------  -----  ----------  -----
' barking'  1.03   ' desk'     -0.87
' puppy'    1.01   ' ledger'   -0.79
' leash'    0.98   ' tax'      -0.79
' kennel'   0.97   ' memo'     -0.75
' bone'     0.97   ' invoice'  -0.75

max    top activating text ([[max token]])
-----  ----------------------------------------------------------------------
45.50  …walked to the park with her[[ dog]] every morning before work…
45.25   The neighbour's[[ dog]] barked at the gate until the post arrived.
45.00  …a bowl of water for the[[ dog]] and a blanket by the door…
```

(Real output can contain odd characters: GPT-2's byte-level tokens are shown as the API returns them.)

### search

Semantic search over explanations, across all models, or within sources of one model
(`--model M --source S`, `--source` repeatable). `-n` rows; `--offset` for the next page.

```
$ npedia search dogs --model gpt2-small --source 6-res-jb -n 4
sim    model       source    index  explanation
-----  ----------  --------  -----  ---------------------------------------
0.590  gpt2-small  6-res-jb  3986   words about dogs
0.550  gpt2-small  6-res-jb  3986   dogs and other household pets
0.520  gpt2-small  6-res-jb  1001   pets and other animals at home
0.510  gpt2-small  6-res-jb  1002   the word "dog" in idioms
showing 4 of 20 results  (more: --offset 20)
```

### topfeat

Top features at each token of your text, for one source (`-n` per token, max 20). With `--all`,
SOURCE is a source set (e.g. `res-jb`) and the strongest features over all its layers are listed.

```
$ npedia topfeat gpt2-small 6-res-jb "The cat sat on the mat" -n 2
source 6-res-jb
pos  token   feature  act    explanation
---  ------  -------  -----  ----------------------------------------------------------------------
1    'The'   1201     32.50  the word "The" starting a sentence
             1202     24.50  the first word of a text
2    ' cat'  1203     49.50  the word "cat"
             1204     16.25  cats and other pets
...

$ npedia topfeat gpt2-small res-jb "The cat sat on the mat" --all -n 3
source     index  max act  at token  explanation
---------  -----  -------  --------  ------------------------------------------------------------
11-res-jb  301    93.50    ' cat'    words about animals
12-res-jb  302    83.50    ' cat'    phrases with the word "cat"
11-res-jb  303    82.50    ' sat'    verbs for sitting or resting, usually followed by a place a…
```

### act

Activation of one feature on each token of your text.

```
$ npedia act gpt2-small 6-res-jb 3986 "My dog barks at the mailman"
gpt2-small/6-res-jb/3986  max 42.50 at token 1
pos  token    act
---  -------  -----
0    'My'     0.00
1    ' dog'   42.50
2    ' bar'   0.00
...
```

### steer

Generates from the prompt with and without adding features (`--feature SOURCE:INDEX:STRENGTH`,
repeatable). Options: `--tokens` (32), `--temperature` (0.5), `--seed` (16), `--multiplier` (4),
`--method SIMPLE_ADDITIVE|ORTHOGONAL_DECOMP`.

```
$ npedia steer gpt2-small "I think that" --feature 6-res-jb:3986:20 --tokens 16
prompt:   'I think that'
default:  ' the weather will be fine tomorrow and we can go out'
steered:  ' my dog would love a walk and a bone, and the puppy too'
share:    https://www.neuronpedia.org/steer/...
note:     Neuronpedia keeps this run, prompt included, under that public link; put nothing private in a steer prompt.
(server-reported remaining steer limit: 238)
```

Each steer run is saved on Neuronpedia, prompt included, under a public share link. Do not put
anything private in a steer prompt.

### lens (J-lens)

Runs the Jacobian lens (default), the logit lens (`--type logit`) or both over a prompt and shows
the top read-out token per position at 8 evenly spaced layers (`--layers 0,20,40` to choose).
`--gen N` generates N tokens (marked `*`); `--chat` sends the prompt as a user turn;
`--top` (1–8) is how many tokens per layer are requested (all are in `--json`).

```
$ npedia lens qwen3.6-27b "The capital of France is" --gen 2
model Qwen/Qwen3.6-27B  top_n 3  prompt tokens 5  generated 2

JACOBIAN_LENS: top-1 read-out per layer (rows = positions, * = generated)
pos  token       L0      L9            L18         L27         L36         L45         L54        L63
---  ----------  ------  ------------  ----------  ----------  ----------  ----------  ---------  ------------
0    'The'       ' The'  ' a'          ' first'    ' same'     ' most'     ' other'    ' new'     ' following'
1    ' capital'  'a'     'isation'     'ization'   ' capital'  ' Capital'  ' capital'  ' city'    ' of'
2    ' of'       ' The'  ' the'        ' of'       ' of'       ' city'     ' city'     ' France'  ' the'
3    ' France'   'e'     ' France'     ' France'   ' France'   ' France'   ' France'   ' is'      ' is'
4    ' is'       ' The'  ' therefore'  ' only'     ' what'     ' cities'   ' Paris'    ' Paris'   ' Paris'
5*   ' Paris'    'ly'    'ian'         ' Paris'    ' Paris'    ' Paris'    ' Paris'    ' Paris'   '.'
6*   '.'         ' The'  ' because'    ' because'  ' which'    ' It'       ' It'       ' It'      '\n\n'
```

## Not covered

Attribution graphs, lists, bookmarks, uploading explanations or vectors, `steer-chat`, the lens
"oracle" and NLA endpoints. Use `--json` and the API docs (<https://www.neuronpedia.org/api-doc>)
for anything beyond the tables.

## Tests

`python3 -m pytest tests/test_npedia.py` runs offline through a fake HTTP opener, against made-up
responses with the same shape as the API's on 2026-11-14: request bodies, headers (User-Agent; the
key only from the environment, only to neuronpedia.org and never printed), caching, rate limiting,
error messages and the tables.
