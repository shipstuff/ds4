# Native Drafter Checkpoint

Date: 2026-06-04

## Goal

Make the native DS4 drafter a practical replacement for the Python/MLX helper, including correctness and speed. Correctness is in good shape. Speed is close to MLX helper parity but not proven better yet.

## Machine Rules

- Do not run heavy validation on the local mini; it has already OOMed.
- CPU-only validation can run on `seslly@192.168.0.60`.
- Metal validation should run on `carl@192.168.0.63:/Users/carl/projects/ds4`.
- Full DS4 target validation should use distributed Metal across `.62/.63`, not the local mini alone.
- Current drafter Python on `.63`: `/Users/carl/projects/anemll-project/env-anemll/bin/python3`.

## Models

- Qwen drafter: `/Users/Shared/models/qwen3.5-0.8b-mlx-4bit`
- DSV4 tokenizer: `/Users/Shared/models/ds4-gguf/dsv4-tokenizer`
- DS4 target: `/Users/Shared/models/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

## Current Perf

Latest retained paired validation on `.63`, after startup logits warmup:

```text
NATIVE_STATS total=10066.873 tokenize=19.937 score=10046.878 align=0.058
PYTHON_STATS total=13267.126 score=9872.526
PARITY max_abs=0.000476621326 mean_abs=2.50853606e-05 n=19
DS4_NATIVE_PARITY_OK
```

Token-rate translation for the 31,198-token long prompt:

- Native score path: about `3105 tok/s`.
- Python/MLX score path: about `3160 tok/s`.
- Gap: about `55 tok/s`, native about `1.7-1.8%` slower.

Latest native-only profile:

```text
NATIVE_STATS total=10070.512 tokenize=13.678 score=10056.793 align=0.041
native drafter profile: tokens=31198 lookahead=4 prefill=9904.196ms argmax=5.677ms lookahead=116.860ms importance=29.975ms select=0.053ms total=10056.778ms
DS4_NATIVE_ONLY_OK
```

The remaining gap is not scalar C overhead. The hot path is GPU/MPS/custom Metal work, mostly prefill.

## Retained Changes

- `ds4_drafter.c`
  - Native tokenizer parity fixes for full-width vertical bar and UTF-8 punctuation-before-letter handling.
  - Stable lookahead=1 keep selection with `default_score_quantum = 2.0e-5`, still overridable by `DS4_DRAFTER_BLOCK_SCORE_QUANTUM`.
  - Startup warmup for tied embedding logits/argmax after preparing Metal caches. This lowered first argmax from about `20ms` to about `5-6ms`.
- `ds4_drafter_metal.m`
  - Batched prefill fixes, including correct `.n_vec` in RoPE args.
  - Faster attention/importance kernels using `fast::exp`.
  - Resident-token full attention improvements: `logits4`, `context4`, `context8`, and default `context16`.
  - Batched importance path, including `importance_reduce_heads` and batch4 logits.
  - Profiling envs retained: `DS4_DRAFTER_METAL_BATCH_PROFILE`, `DS4_DRAFTER_METAL_ATTENTION_PROFILE`, `DS4_DRAFTER_METAL_TOKEN_PROFILE`.
- `speed-bench/ds4_live_drafter.py`
  - Python helper imports installed `mlx_lm`; native backend does not depend on local `mlx-lm`.
  - Debug envs retained for token/keep/argmax inspection.
- `speed-bench/specprefill_native_parity.py`
  - Canonical parity harness location.
  - Includes native warmups, token hash debug, and Python timing output.

## Wiring

- `Makefile` links native drafter objects into `ds4` and `ds4-server`.
- On Darwin, `DRAFTER_OBJS = ds4_drafter.o ds4_drafter_metal.o`.
- CPU target uses `ds4_drafter_cpu.o`.
- `speed-bench/specprefill_chat_loop.py` defaults to `speed-bench/ds4_live_drafter.py`.
- Removed stale root-level helper copies; helper scripts now live under `speed-bench/`.

## Rejected Or No-Win Experiments

- Reverted fused resident-token argmax; it was slower.
- `DS4_DRAFTER_MPS_CAUSAL_ATTENTION_BLOCK_ROWS`: `768`, `896`, `1152`, `1280`, and earlier values did not beat current `1024`.
- `DS4_DRAFTER_MPS_CAUSAL_SOFTMAX=1`: parity passes but did not close the gap.
- `DS4_DRAFTER_LINEAR_DELTA_THREADS`: `16`, `64`, `128` were slower than current `32`.
- Same-process `--native-warmups`: improved argmax/lookahead but made prefill worse overall.
- Other rejected paths: token matmul, attention batch KV, all-KV pack, direct KV, direct-head attention, MPSGraph SDPA, online softmax, MLP pair, QKVZ pair, tiny ops, prepacked K, softmax thread sweeps, F16 weights off, transposed weights off, quant tile variants, private scratch, prepare warmup off, and MPS matmul off.

## Bottleneck Notes

Representative batch profile before the latest startup argmax warmup:

```text
input_norm ~= 5.8ms
linear_proj ~= 1404ms
linear_conv ~= 464-476ms
linear_scan ~= 630-635ms
linear_out ~= 436ms
full_proj ~= 284ms
full_rope ~= 44ms
full_attn ~= 3947ms
full_out ~= 125ms
mlp_proj ~= 1572ms
mlp_down ~= 966ms
```

Full-attention detail per full layer:

```text
kv_pack ~= 1ms
q_pack ~= 10ms
qk ~= 302ms
softmax ~= 77ms
pv ~= 301ms
gate ~= 11ms
```

Next likely optimization work is native prefill QK/PV and affine/MLP layout or MPS scheduling. Do not move work out of `score_ms` just to improve accounting; the target is real chatloop wall time.

## Resume Commands

Build locally only:

```sh
cd /Users/carl/projects/ds4
make ds4
```

Sync and build on `.63`:

```sh
cd /Users/carl/projects/ds4
rsync -az ds4_drafter.c ds4_drafter_metal.m ds4_drafter_metal.h speed-bench/ds4_live_drafter.py speed-bench/specprefill_native_parity.py carl@192.168.0.63:/Users/carl/projects/ds4/ --relative
ssh carl@192.168.0.63 'cd /Users/carl/projects/ds4 && make ds4'
```

Paired parity on `.63`:

```sh
ssh carl@192.168.0.63 'cd /Users/carl/projects/ds4 && /Users/carl/projects/anemll-project/env-anemll/bin/python3 speed-bench/specprefill_native_parity.py --repo /Users/carl/projects/ds4 --model /Users/Shared/models/qwen3.5-0.8b-mlx-4bit --dsv4-tokenizer /Users/Shared/models/ds4-gguf/dsv4-tokenizer --mlx-lm /Users/carl/projects/mlx-lm --drafter-python /Users/carl/projects/anemll-project/env-anemll/bin/python3 --native-metal --text-file tests/long_context_story_prompt.txt --spans 19 --lookahead 4 --pool-kernel 13 --block-size 128 --keep-fraction 0.125 --sink-size 128 --tail-keep 128 --tolerance 0.01'
```

Native-only profile on `.63`:

```sh
ssh carl@192.168.0.63 'cd /Users/carl/projects/ds4 && DS4_DRAFTER_PROFILE=1 /Users/carl/projects/anemll-project/env-anemll/bin/python3 speed-bench/specprefill_native_parity.py --repo /Users/carl/projects/ds4 --model /Users/Shared/models/qwen3.5-0.8b-mlx-4bit --dsv4-tokenizer /Users/Shared/models/ds4-gguf/dsv4-tokenizer --native-only --native-metal --text-file tests/long_context_story_prompt.txt --spans 19 --lookahead 4 --pool-kernel 13 --block-size 128 --keep-fraction 0.125 --sink-size 128 --tail-keep 128 --tolerance 0.01'
```

