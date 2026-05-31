#!/usr/bin/env python3
"""Resident live drafter scorer for DS4 SpecPrefill.

Protocol on stdin/stdout:
  SCORE <expected_ds4_tokens> <utf8_byte_len>\n
  <rendered transcript bytes>

Response:
  OK <n_scores> <total_ms> <tokenize_ms> <score_ms> <align_ms>\n
  <n_scores little-endian float32 values>

This process is intentionally persistent: the MLX/Qwen drafter is loaded once
and reused across DS4 chat turns.
"""

from __future__ import annotations

import argparse
import array
import bisect
import sys
import time
from pathlib import Path


def realign_to_dsv4(scorer_scores, scorer_offsets, dsv4_offsets):
    out = []
    scorer_starts = [s for s, _ in scorer_offsets]
    scorer_ends = [e for _, e in scorer_offsets]
    fallback = sum(scorer_scores) / max(1, len(scorer_scores))
    for a, b in dsv4_offsets:
        if a == b:
            out.append(fallback)
            continue
        j = bisect.bisect_right(scorer_ends, a)
        num = 0.0
        den = 0
        k = max(0, j - 1)
        while k < len(scorer_offsets) and scorer_starts[k] < b:
            ov = max(0, min(b, scorer_ends[k]) - max(a, scorer_starts[k]))
            if ov > 0:
                num += scorer_scores[k] * ov
                den += ov
            k += 1
        out.append(num / den if den > 0 else fallback)
    return out


def normalize(scores):
    if not scores:
        return scores
    smin = min(scores)
    smax = max(scores)
    span = smax - smin
    if span <= 0:
        return [0.0 for _ in scores]
    return [(s - smin) / span for s in scores]


class FakeScorer:
    def score(self, text: str, expected: int):
        # Test-only deterministic shape: later tokens score slightly higher.
        del text
        if expected <= 0:
            return [], (0.0, 0.0, 0.0)
        return [i / max(1, expected - 1) for i in range(expected)], (0.0, 0.0, 0.0)


class MlxScorer:
    def __init__(self, args):
        sys.path.insert(0, args.mlx_lm_dir)
        sys.path.insert(0, args.specprefill_lib)

        import mlx.core as mx  # noqa: PLC0415
        from mlx_lm import load  # noqa: PLC0415
        from mlx_lm.spec_prefill import compute_keep_indices  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.mx = mx
        self.compute_keep_indices = compute_keep_indices
        self.n_lookahead = args.n_lookahead
        self.pool_kernel = args.pool_kernel
        self.prefill_step_size = args.prefill_step_size

        print(f"ds4-live-drafter: load scorer tokenizer {args.scorer_model}", file=sys.stderr, flush=True)
        self.scorer_tk = AutoTokenizer.from_pretrained(args.scorer_model, trust_remote_code=True)
        print(f"ds4-live-drafter: load dsv4 tokenizer {args.dsv4_tokenizer}", file=sys.stderr, flush=True)
        self.dsv4_tk = AutoTokenizer.from_pretrained(args.dsv4_tokenizer, trust_remote_code=True)
        print(f"ds4-live-drafter: load MLX scorer model {args.scorer_model}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        self.model, _ = load(args.scorer_model)
        mx.eval(self.model.parameters())
        print(f"ds4-live-drafter: ready in {time.perf_counter() - t0:.2f}s", file=sys.stderr, flush=True)

    def score(self, text: str, expected: int):
        t0 = time.perf_counter()
        scorer_enc = self.scorer_tk(text, return_offsets_mapping=True, add_special_tokens=False)
        dsv4_enc = self.dsv4_tk(text, return_offsets_mapping=True, add_special_tokens=False)
        scorer_ids = scorer_enc["input_ids"]
        scorer_offsets = list(scorer_enc["offset_mapping"])
        dsv4_offsets = list(dsv4_enc["offset_mapping"])
        t1 = time.perf_counter()

        prompt = self.mx.array(scorer_ids, dtype=self.mx.uint32)
        keep = self.compute_keep_indices(
            self.model,
            prompt,
            lookahead_steps=self.n_lookahead,
            pool_kernel=self.pool_kernel,
            block_size=32,
            keep_fraction=0.3,
            prefill_step_size=self.prefill_step_size,
            sink_size=16,
            tail_keep=256,
            aggregation="pool_then_max",
        )
        self.mx.eval(keep)
        scorer_scores = [0.0] * len(scorer_ids)
        for i in keep.tolist():
            if 0 <= int(i) < len(scorer_scores):
                scorer_scores[int(i)] = 1.0
        t2 = time.perf_counter()

        dsv4_scores = normalize(realign_to_dsv4(scorer_scores, scorer_offsets, dsv4_offsets))
        if len(dsv4_scores) != expected:
            raise ValueError(
                f"dsv4 tokenizer produced {len(dsv4_scores)} tokens, expected DS4 transcript has {expected}"
            )
        t3 = time.perf_counter()
        return dsv4_scores, ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0)


def send_error(msg: str) -> None:
    sys.stdout.buffer.write(f"ERR {msg}\n".encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def send_scores(scores, timings) -> None:
    tokenize_ms, score_ms, align_ms = timings
    total_ms = tokenize_ms + score_ms + align_ms
    sys.stdout.buffer.write(
        f"OK {len(scores)} {total_ms:.3f} {tokenize_ms:.3f} {score_ms:.3f} {align_ms:.3f}\n".encode("ascii")
    )
    arr = array.array("f", scores)
    if sys.byteorder != "little":
        arr.byteswap()
    sys.stdout.buffer.write(arr.tobytes())
    sys.stdout.buffer.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scorer-model", required=True)
    ap.add_argument("--dsv4-tokenizer", default="/Users/Shared/models/ds4-gguf/dsv4-tokenizer")
    ap.add_argument("--specprefill-lib", default="/Users/carl/projects/anemll-project/scripts/heterogeneous")
    ap.add_argument("--mlx-lm-dir", default="/Users/carl/projects/mlx-lm")
    ap.add_argument("--n-lookahead", type=int, default=4)
    ap.add_argument("--pool-kernel", type=int, default=13)
    ap.add_argument("--prefill-step-size", type=int, default=2048)
    args = ap.parse_args()

    scorer = FakeScorer() if args.scorer_model == "__fake__" else MlxScorer(args)
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        line_s = line.decode("ascii", errors="replace").strip()
        if line_s == "QUIT":
            return 0
        try:
            op, expected_s, nbytes_s = line_s.split()
            if op != "SCORE":
                raise ValueError(f"unknown op {op}")
            expected = int(expected_s)
            nbytes = int(nbytes_s)
            payload = stdin.read(nbytes)
            if len(payload) != nbytes:
                raise ValueError(f"short prompt payload {len(payload)}/{nbytes}")
            text = payload.decode("utf-8")
            scores, timings = scorer.score(text, expected)
            send_scores(scores, timings)
        except Exception as exc:  # keep the resident helper alive for debuggability
            send_error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
