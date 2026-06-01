## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

## SpecPrefill Chat-Loop Comparison

The baseline vs SpecPrefill numbers should come from the chat-loop harness, not
from fixed one-shot prompts. The harness keeps the model process resident and
uses the same cold-full-prompt semantics as the mini-01 Qwen validation:
baseline and drafter both prefill the full accumulated transcript on each turn.

First download the target model, DSV4 tokenizer, and the MLX drafter:

```
./download_model.sh specprefill
```

If the DS4 GGUF is already present locally and you only need the tokenizer and
drafter:

```
./download_model.sh drafter
```

### Single-Node Interactive Test

On a single machine with enough unified memory, roughly >96 GB, run the target
model and live drafter locally and use the normal `ds4>` prompt. Do not use
`--spec-prefill-self-score` for this test; that path scores with the target model
and is much slower than the resident MLX drafter path we validated.

With the default download paths:

```
make ds4

export PYTHONPATH="$HOME/projects/mlx-lm:${PYTHONPATH:-}"

./ds4 -m ./ds4flash.gguf \
  --ctx 65536 \
  --spec-prefill=0.3 \
  --spec-prefill-tail 256 \
  --spec-prefill-chunk 32 \
  --spec-prefill-cache fresh \
  --spec-prefill-drafter-model ./gguf/qwen3.5-0.8b-mlx-4bit \
  --spec-prefill-drafter-python python3 \
  --spec-prefill-drafter-script speed-bench/ds4_live_drafter.py \
  --spec-prefill-drafter-tokenizer ./gguf/dsv4-tokenizer
```

That lands at the normal `ds4>` prompt. Every turn, `fresh` mode has the live
drafter score the full true transcript, then DS4 compresses and prefills the
selected context. This is the interactive path to use for manual validation.

Then build DS4 and launch the comparison in tmux:

```
make ds4
speed-bench/run_chatloop_comparison_tmux.sh
tmux attach -t ds4-chatloop-compare
```

The launcher writes:

```
/tmp/ds4_chatloop_compare/metrics.csv
/tmp/ds4_chatloop_compare/chat_baseline_vs_specprefill.png
/tmp/ds4_chatloop_compare/chat_baseline_vs_specprefill.md
```

Useful overrides:

```
CTX=65536 N_PREDICT=1024 OUT_DIR=/tmp/ds4_m5_max_128gb \
  speed-bench/run_chatloop_comparison_tmux.sh
```

For a distributed coordinator, start workers separately, then run:

```
DIST=1 COORD_LAYERS=0:19 COORD_LISTEN_HOST=0.0.0.0 COORD_LISTEN_PORT=1234 \
  DIST_PREFILL_CHUNK=1024 speed-bench/run_chatloop_comparison_tmux.sh
```

The SpecPrefill mode uses the live local drafter by default:

```
./gguf/qwen3.5-0.8b-mlx-4bit
```

Override it with `DRAFTER_MODEL=/path/to/mlx-drafter`. The DS4 target model,
DSV4 tokenizer, drafter Python, and MLX checkout can be overridden with `MODEL`,
`DRAFTER_TOKENIZER`, `DRAFTER_PYTHON`, and `MLX_LM_DIR`.

### Server Mode

`ds4-server` accepts the same live-drafter SpecPrefill options for cold HTTP
request prefills:

```
export PYTHONPATH="$PWD/../mlx-lm:${PYTHONPATH:-}"

./ds4-server -m ./ds4flash.gguf \
  --ctx 65536 \
  --host 127.0.0.1 \
  --port 8000 \
  --spec-prefill=0.3 \
  --spec-prefill-tail 256 \
  --spec-prefill-chunk 32 \
  --spec-prefill-cache fresh \
  --spec-prefill-drafter-model ./gguf/qwen3.5-0.8b-mlx-4bit \
  --spec-prefill-drafter-python python3 \
  --spec-prefill-drafter-script speed-bench/ds4_live_drafter.py \
  --spec-prefill-drafter-tokenizer ./gguf/dsv4-tokenizer
```

Server SpecPrefill currently supports fresh compression. Existing live and disk
KV cache hits still run through the normal server cache path. Tool-enabled
requests use protected spans so tool schemas, tool-call IDs, tool results, and
file contents stay exact while surrounding natural-language history can still be
compressed.
