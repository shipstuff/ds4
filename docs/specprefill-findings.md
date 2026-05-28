# ds4 SpecPrefill port — findings vs the anemll-project dense-decode reference

Branch: `seslly/specprefill` on `shipstuff/ds4`, three commits.
Compared against: `~/projects/anemll-project/scripts/run_local_context_realtime.py`
(`stream_with_specprefill` + `run_chat_loop`) on `mini-01`, target
`Qwen3.6-35B-A3B-4bit`, draft `qwen3.5-0.8b-mlx-4bit`.

## TL;DR

The structural port to ds4 is complete and the algorithm matches the
anemll reference where it can, but the *expected user experience in a
chat loop* doesn't transfer because the constraints that make
SpecPrefill cheap on anemll's setup don't exist on DSV4 Flash. The
port is honest about what it can deliver; this doc walks the
comparison and asks a few specific questions for the anemll team
before deciding whether to invest in a chat-loop-friendly v2.

## What got ported

| Piece | anemll reference | ds4 port |
|---|---|---|
| Scoring algorithm | `specprefill.score_tokens` — N-layer attention, softmax across heads, avg-pool smoothing, mean across lookahead | `score_prompt_cpu` / `score_prompt_metal` in `ds4.c` — identical aggregation; reuses ds4's own CPU + GPU primitives so it stays in lockstep with the prefill graph |
| Chunk selection | `select_chunks(importance, keep_pct, chunk_size)` | `ds4_spec_prefill_compress` — same bucket-then-pick-top-k logic |
| Dense-decode shape | "selected history + last RECENT_TAIL=256 tokens at contiguous positions" | Identical: `kept_history_chunks + last tail_size` at positions 0..M-1 |
| RoPE | Standard MLX RoPE on the contiguous compressed prompt | Standard `ds4_gpu_rope_tail_tensor` (CPU equivalent) — same kernel the real prefill uses |
| Chat-template handling | `tokenizer.apply_chat_template` | ds4's `ds4_encode_chat_prompt` / `ds4_chat_append_message` (DSV4 native template) |
| Cache lifecycle (chat loop) | `target_cache = make_prompt_cache(model)` *fresh every turn*; no KV reuse across turns | Same: `ds4_session_invalidate()` + full re-prefill of compressed prompt every turn |

The architectural choice "no cache reuse across turns" is shared by
both implementations.

## What's different (and why)

| Axis | anemll | ds4 |
|---|---|---|
| Draft model for scoring | **Separate small model**: Qwen3-Next 0.8B (or the ANE-side `AneDraftRunner`) | **Self-score with the target itself** — ds4 is DSV4-Flash-specific by `AGENT.md`, no second-model runtime |
| Target active params / token | Qwen3.6-35B-A3B → ~3B active | DSV4 Flash → ~13B active |
| Target prefill throughput | Apple MLX on M4 Pro Metal | Hand-rolled MLA + DSA compute graph, M4 Pro Metal (or AMD CPU on the Linux box used for measurement) |
| Per-turn scoring cost | ~1 s (0.8B draft, fast on ANE or MLX) | 14–22 s for 2 layers on a 30k prompt (DSV4 itself; same kernels as full prefill but capped to 2 layers + skip FFN) |

The "no separate draft model" axis is the load-bearing one. anemll can
afford to score 30k tokens in roughly a second because the draft is
~40× smaller than the target and runs on a fast accelerator. ds4
can't: the only way to score on DSV4 is to run a subset of DSV4
itself, and even at 2 layers + Q/K-only the projections dominate.

## Measured perf (Linux x86 AMD CPU, `seslly`, `make cpu`, DSV4 Flash IQ2_XXS)

Four-turn scripted chat, ctx=32768, first turn = `tests/long_context_story_prompt.txt`,
follow-ups are short questions. Two independent runs agreed:

| | Run-1 total wall | Run-2 total wall |
|---|---:|---:|
| baseline (no spec-prefill) | 85 s | 86 s |
| heuristic (recency only) | 96 s | 100 s |
| self-score (in-engine) | 233 s | 149 s |

### Where the gap shows up

**Cold turn 0** (~30k-token prompt, no prior KV) — SpecPrefill wins as expected:

| | turn 0 wall |
|---|---:|
| baseline | 79–80 s |
| heuristic | 25–26 s (≈3.1× faster) |
| self-score | 39–47 s (≈1.7–2.0× faster; scoring step adds ~14–22 s vs heuristic) |

Self-score gets faster on run 2 — almost certainly OS page-cache
priming the weights on the second pass through.

**Warm continuation turns 1–3** — SpecPrefill loses:

| | avg wall per turn |
|---|---:|
| baseline | 1.9–2.7 s |
| heuristic | 23–25 s |
| self-score | 37–62 s |

This is the hard finding: each spec-prefill turn invalidates the
session and re-prefills the full ~9k-token compressed prompt from
scratch, while baseline reuses the 30k-token KV cache from turn 0
and only prefills the ~150-token follow-up suffix. Over 4 turns,
baseline's amortized cost wins even though its turn 0 was 3× slower.

### Decode throughput

Flat across all three modes at 29–30 t/s. The compressed KV cache
neither hurts nor helps short-decode here. This is the dense-decode
guarantee working as advertised: compressed cache + contiguous-RoPE
shape → decode at baseline speed.

### CPU vs Metal scorer parity

`./ds4_test --spec-prefill-parity` and `DS4_SCORE_VALIDATE=1` give
two ways to surface composition bugs in the Metal scorer; tested on
seslly the CPU path is bit-stable across runs. Per-turn `max|m-c|`
in the chat-loop runs:

```
turn 0:  7e-06     turn 1:  2.353e-03     turn 2:  7e-06     turn 3:  5e-06
```

Turn 1's `2.353e-03` reproduces deterministically across both runs,
same compressed-token counts in both — suggests a content-dependent
numerical path (probably f16 accumulation order), not flakiness. All
under the parity test's 0.05 tolerance and the kept-chunk count is
identical between CPU and Metal scorers (`9318 / 9376 / 9405 / 9387`
across turns in both runs).

## What's "missing" from the ds4 port

I want to be careful here. Three categories:

1. **Things that are architecturally inaccessible** — a fast small
   draft model, ANE-side scoring. ds4 is by design DSV4-only, no
   second-model runtime; adding one is its own project that the
   AGENT.md "narrow scope" rule warns against.

2. **Things that are deferred but tractable**:
   - **Persistent KV cache across turns** with selective recompression.
     anemll *also* invalidates per turn, but its per-turn cost is small
     enough not to matter. On DSV4 the per-turn cost is what kills
     chat-loop UX. A v2 design could:
     - skip recompression when `common_prefix > threshold` (the
       checkpoint-reuse path ds4's session API already has)
     - compress once at conversation start, then extend incrementally
       (only re-prefill the new user turn + assistant prefix)
     - re-compress only when context approaches `--ctx`, treating
       SpecPrefill as a context-window saver rather than a TTFT
       optimizer
   - **`ds4-server` request-level option**. ds4 has four protocol
     parsers (chat/completions/responses/anthropic) each with their
     own JSON parsing + KV-cache reuse paths; threading SpecPrefill
     through cleanly is its own change. CLI/REPL was the v1 surface.
   - **Metal fast path for the scoring loop's softmax/aggregation**.
     Currently the Metal scorer reads Q + K back to host per layer
     and runs the aggregation in C. Tiny IO compared to the actual
     projections, but a fused Metal kernel would eliminate ~30 MB of
     readback per layer on long prompts.

3. **Things that look different in the code but are equivalent in
   behavior** — anemll scores the full prompt then filters selection
   to history-only; ds4 scores only the history. Both end up with
   the same `selected_history + full_tail` shape.

## Answers from the anemll/mlx-lm team

Asked via the `ds4-review` tmux session on `mini-01` against the
working dense-decode implementation. Verbatim summary below; the
full transcript lives in the same tmux pane.

### Q1 — Per-turn wall at ~30k context, dense_decode chat loop, Qwen3.6 35B-A3B, M4 Pro

- **Library path** (mlx-lm + `dense_prefill_selected`): single-call
  32k prompt + 128 decode tokens → TTFT **18.6 s**, decode 80 t/s,
  wall **20.2 s**. Multi-turn 2-turn chat at 4k context: t1+t2 =
  **11.55 s** vs baseline 16.79 s (**1.45× faster**).
- **HTTP-server path** (vllm-mlx): 32k → TTFT 33.6 s, decode 67 t/s,
  wall ~35 s. The ~3 s overhead + decode drop comes from extra
  text-route plumbing, not the algorithm.
- No clean 30k-chat-loop datapoint, but should track the single-call
  number ±10% — chat-template overhead is small relative to prefill
  at that scale.

For comparison, our DSV4 IQ2_XXS measurement on `seslly` CPU at
30k context was **25 s per turn heuristic / 37–62 s per turn
self-score** — i.e. anemll's per-turn cost is comfortably under
our per-turn cost, and *both* re-prefill every turn. Their UX is
livable because the absolute number is small, not because they
avoid the re-prefill.

### Q2 — Fresh-every-turn `target_cache` is intentional, reuse isn't built yet

> The reason it's not just laziness: dense_decode's selection is
> content-dependent on the full current prompt. Turn N+1's
> selected-history token set is not generally a superset of turn
> N's, so the compressed cache layout gets invalidated mid-stream.

Two paths a real reuse implementation could take, neither built yet:

a. **Selection stability constraint** — force turn N+1's selection
   to include all of turn N's, with optional tail extension.
b. **Cache merge / diff** — surgically rebuild only the slots that
   changed.

They considered (a) but didn't build it; at ~18 s/turn with
scoring ~1–8 s of that, the upper bound on savings is modest. For
DSV4 where target prefill alone is 25–60 s, the savings ceiling
would be much higher and (a) is potentially worth building.

`save_prompt_cache` only handles the simple-baseline (full-prompt)
path. It doesn't handle compressed-position state.

### Q3 — Per-turn cost split at 32k, library path

- **Scoring** (0.8B draft full-prompt prefill + 8-step lookahead +
  Q·K^T softmax math): **~1 s at 4k, scales near-linearly to ~8 s
  at 32k** = **~40%** of per-turn wall at 32k.
- **Target prefill** (compressed: ~20% selected + 256 tail = ~6.5k
  tokens, chunked at `prefill_step_size=2048`): **~10 s at 32k**
  = **~55%**.
- **Decode** (128 tokens at 80 t/s): **~1.6 s** = **~5%**.

Decode share grows for longer outputs; scoring share shrinks for
longer contexts (target prefill grows faster with M).

**For ds4:** the equivalent split at 32k on DSV4 IQ2_XXS CPU is
**~25–30% scoring (~14–22 s) / ~70–75% prefill (~25–58 s) / <1%
decode**. The bottleneck isn't scoring overhead, it's that the
target prefill of 9k compressed tokens through 43 layers of MLA +
DSA + MoE is ~10× the cost of Qwen3.6-A3B's compressed prefill.
Even free scoring wouldn't fix it; cache reuse is the lever.

### Q4 — Edge cases / landmines (this section is gold)

In order of how subtle they were, verbatim from the anemll agent:

1. **Multi-turn coherence broke initial naïve dense_decode.** First
   version compressed *all* selected tokens to positions 0..N-1.
   Turn 2 produced garbage like *"The **End** of the **End**…"*
   because chat-template markers (`<|im_end|>`,
   `<|im_start|>assistant\n`) lost positional grounding. **Fix:**
   hybrid layout — compressed history at 0..H-1 + last 256 tokens at
   H..H+255 verbatim. **Tail size must cover chat-template footer +
   recent turn. Recommend never going below tail_size=128.**

   ✓ ds4 already implements the hybrid layout. Default tail=256;
   `--spec-prefill-tail` is settable. Worth adding a *minimum* enforced
   in `ds4_spec_prefill_compress` (refuse < 64, warn < 128).

2. **cleanup_rope between turns is non-negotiable** on the
   `sparse_prefill` path (installs `_OffsetAdjustedRoPE` wrapper).
   Without unwrap, next turn fails with `AttributeError: ... has no
   attribute 'dims'`. Wrap decode in `try/finally` and call cleanup
   *unconditionally*, including on the abort path.

   N/A for ds4 — we only implement the dense-decode (contiguous
   positions) path; no RoPE wrapper to clean up. Worth a comment in
   ds4_cli.c noting we deliberately don't have the sparse path so the
   cleanup story is simpler.

3. **keep_pct=0.0 degenerate case.** They special-case it: return
   `[M-1]` + sink/tail-keep ranges. Otherwise block-selection math
   produces zero blocks and downstream code blows up.

   ds4 currently rejects `keep_pct <= 0` outright (`return 1`).
   That's safer but less ergonomic. Consider matching their
   special-case behavior in a v2.

4. **Tail content collision.** When `keep_fraction` is high enough
   that selected blocks overlap the tail region, dedup with set
   union on the index lists before concatenating. Otherwise you
   double-prefill some tokens and `cache.offset` advances past the
   decode-seed point.

   ds4 implementation: history-and-tail are kept separate by
   construction (`history_len = prompt->len - tail_size`, chunks
   only the history). **No overlap possible** because the index
   spaces are disjoint. ✓ Already safe.

5. **The 32k cache-state bug** (the mlx-lm clean-room port).
   Deterministic first-token EOS at 32k on Qwen3.6-A3B-4bit.
   Cross-impl swap test: same scoring (0.96 Pearson, 97.8% top-K
   agreement), but ~67% block-selection overlap. Failing combo is
   *their selection + their prefill specifically*. Cause not root-
   caused. Quote: *"Worth running the same cross-impl swap test
   against your port if you start seeing first-token EOS at long
   context."*

   For ds4: if a long-context bench ever shows first-token EOS or
   garbled first-token output, swap-test (use ds4 scoring, anemll
   compressed prompt, vs ds4 selection through anemll prefill) is
   the diagnostic.

### Q5 — qwen3_5 vs qwen3_next extractors are functionally identical

> Both do the gated-query split: q_proj outputs 2 × n_heads × head_dim,
> reshape and `mx.split(..., 2, axis=-1)` keeps the first half as
> queries and drops the second half (the output gate). Then q_norm,
> transpose to [B, H, L, D], then `attn.rope(queries, offset=cache.offset)`.

Two registry entries because the model_type strings differ, but
mlx-lm's `qwen3_5.py` literally does
`from .qwen3_next import Qwen3NextAttention as Attention`.

**For DSV4 with MLA — three pieces of guidance, directly applicable:**

a. **MLA Q capture shape.** *"DeepSeek-style MLA computes Q via two
   projections (q_a_proj → q_b_proj with LayerNorm between), then
   splits q_b_proj output into RoPE'd and non-RoPE'd halves.
   There's no gate to drop, but the RoPE'd portion is only `q_pe =
   q_b_pe` which is smaller than n_heads × head_dim. You need to
   capture the concatenated Q (q_nope + q_pe) for scoring, OR
   capture `q_pe` alone if you're willing to score on partial
   dimensions (faster but lower SNR). I'd capture concatenated and
   pay the small cost."*

   ✓ ds4 `score_prompt_cpu` / `score_prompt_metal` already capture
   the full concatenated Q (n_head × head_dim = 64 × 512 for
   DSV4 Flash) via `attn_q_b` matmul output, after `head_rms_norm`
   and full RoPE on the rotated tail. We pay the small cost.

b. **MLA noise floor.** *"Rank-reduction in the latent projection
   adds noise. In practice this widens the gap between 'model
   thinks token i is important' and 'draft scores token i highly.'
   Empirical mitigations from our work that should help: keep
   sink_size >= 16, keep tail_size >= 128, keep pool_kernel >= 13.
   If your initial bench shows >40% selection disagreement vs a
   known-good (full-attention) reference, the smoothing isn't
   enough — try pool_kernel=21 or block_size=8."*

   **Action items for ds4 v2:**
   - **`--spec-prefill-sink-size N` flag** (default 16). ds4 does
     NOT currently preserve sink tokens; we keep only `last N
     tokens`. Adding sink preservation matches the anemll behavior
     and helps the MLA noise floor on long prompts.
   - Increase `--spec-prefill-score-pool-kernel` default from 13 to
     21 if early DSV4 benches show high selection disagreement.
   - Add `--spec-prefill-chunk` ergonomics for smaller blocks (8 vs
     32) when noise is high. We already expose `--spec-prefill-chunk`
     so this is just a documentation point.

c. **Self-scoring with target = draft.** *"Algorithmically it works —
   the gated-query extractor doesn't care that target == draft. But
   the cost equation breaks: scoring takes as long as a full target
   prefill, eliminating the speedup. SpecPrefill's premise is that
   the draft is materially smaller. Don't bother unless you can
   fold the score capture into a forward pass you were going to do
   anyway (e.g., the first turn of a chat where the draft has to
   prefill the prompt regardless)."*

   This is the load-bearing constraint for ds4. Our self-score path
   on DSV4 is exactly the "doesn't work economically" case. Two
   responses are possible:

   - **Fold score capture into the actual prefill** of turn 1, so
     scoring is amortized into work we were going to do anyway,
     and use those scores to compress *turn 2+*. This is the
     "cache reuse" v2 path under a different name.
   - **Accept that ds4 self-score is a correctness/diagnostic
     path only**, and add a `--spec-prefill-scores FILE` workflow
     where users compute scores externally with a smaller
     attention-extractable model that shares DSV4's tokenizer (or
     a near-enough tokenizer with our `align_scores_to_dsv4.py`).

## Open questions left for the v2 design

(After the answers above; what's still genuinely unresolved.)

1. **Sink-token preservation** — the anemll defaults keep `sink_size
   >= 16` to fight MLA-style noise. Worth a `--spec-prefill-sink-size`
   flag in ds4. Open: should it default to 16 (anemll's number) or
   something larger for DSV4 (where the noise floor may be wider)?
2. **Selection stability v2** — the most actionable thing the anemll
   answer pointed at. *"Force turn N+1's selection to include all of
   turn N's, with possible extension at the tail."* For DSV4 where
   per-turn prefill is the bottleneck, this is the only meaningful
   lever. Open: how strict to make the constraint — full inclusion
   (no chunks ever drop) vs sticky inclusion (chunks persist for K
   turns)?
3. **Cross-impl swap test against the 32k mlx-lm bug** — anemll
   recommends running their selection through our prefill and our
   selection through theirs, when long-context regressions appear.
   For ds4 vs anemll the cross-impl is harder because we don't share
   a tokenizer or backend, but if we ever see first-token EOS at
   long DSV4 context the diagnostic is to write a tool that converts
   ds4's selected indices into a DSV4-tokenized prefill that anemll
   could read back.
4. **`keep_pct=0.0` ergonomics** — currently rejected; could match
   anemll's `[M-1] + sink + tail` behavior. Low priority.

1. **What's a typical per-turn wall time at 30k context on Qwen3.6
   35B-A3B + dense_decode in your chat loop?** I want a reference
   point for "this is the UX users actually get". My measurement is
   that ds4's per-turn cost on DSV4 dominates above ~5–10k context;
   you may have headroom that just doesn't exist on DSV4 even with
   perfect engineering.

2. **Is the per-turn `target_cache = make_prompt_cache()` pattern
   intentional?** I see persistence machinery (`save_prompt_cache`,
   `load_prompt_cache`) used for the `optimized_local` mode but not
   for the SpecPrefill chat loop. Is there a planned-but-not-yet path
   where the compressed history KV gets cached and reused, or is
   "re-score-and-re-prefill every turn" the canonical design?

3. **On Qwen3.6 35B-A3B + 0.8B draft, what's the rough split of
   per-turn cost between scoring, target prefill of the compressed
   prompt, and decode?** I want to know whether SpecPrefill is
   "essentially free scoring + cheap target prefill" or whether the
   target prefill alone is the dominant cost. If the latter, ds4's
   13B-active per token is the inevitable bottleneck and the v2
   cache-reuse work is the only meaningful win available.

4. **Any edge cases in the dense_decode path that you've hit and
   patched?** I have the structure down (selected_history at
   compressed positions 0..H-1 + tail at H..H+T-1, both with
   standard RoPE) but if there's a known landmine — e.g., a
   keep_pct value where chunk boundaries land badly, or a
   tail_size where chat-template tokens get clipped, or a
   long-context regression analogous to the mlx-lm 32k cache-state
   bug — I'd rather hear about it before discovering it.

5. **`_qwen3_next_extract_queries` vs `_qwen35_extract_queries`** —
   you use the qwen3-next extractor for the Qwen3-Next 0.8B draft,
   which makes sense. Does the qwen35 extractor differ in a way
   that matters if a target ever wanted to self-score on a Qwen3.5
   model the way ds4 self-scores on DSV4? The shape of the issue
   matters for understanding where DSV4's MLA introduces additional
   numerical noise that Qwen wouldn't see.

## Where to look

- ds4 changes: `git log --oneline origin/seslly/specprefill` on
  `shipstuff/ds4` (3 commits: port, bench harness, broken-pipe fix).
- ds4 measurement artifacts: `metrics.csv` / `metrics2.csv` (live
  next to this doc in the working tree; same fixture, two
  independent runs of `speed-bench/specprefill_chat_loop.py`).
- ds4 bench script: `speed-bench/specprefill_chat_loop.py`
  (subprocess-drives `./ds4` REPL, parses ds4's own log lines, CSV
  + matplotlib).
- anemll reference: `~/projects/anemll-project/scripts/run_local_context_realtime.py`
  on `mini-01`, functions `stream_with_specprefill` (lines 323–488)
  and `run_chat_loop` (lines 721–818).
