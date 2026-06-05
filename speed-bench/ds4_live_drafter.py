#!/usr/bin/env python3
"""Resident live drafter scorer for DS4 SpecPrefill.

Protocol on stdin/stdout:
  SCORE <expected_ds4_tokens> <utf8_byte_len>\n
  <rendered transcript bytes>

  SCORE2 <expected_ds4_tokens> <utf8_byte_len>\n
  <rendered transcript bytes>
  <expected_ds4_tokens pairs of little-endian uint32 byte start/end offsets>

Response:
  OK <n_scores> <total_ms> <tokenize_ms> <score_ms> <align_ms>\n
  <n_scores little-endian float32 values>

This process is intentionally persistent: the drafter is loaded once and reused
across DS4 chat turns.
"""

from __future__ import annotations

import argparse
import array
import bisect
import math
import os
import struct
import sys
import time


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


def byte_spans_to_char_spans(text: str, byte_spans):
    byte_len = len(text.encode("utf-8"))
    byte_to_char = [0] * (byte_len + 1)
    pos = 0
    for i, ch in enumerate(text):
        n = len(ch.encode("utf-8"))
        for j in range(n):
            byte_to_char[pos + j] = i
        pos += n
        byte_to_char[pos] = i + 1
    out = []
    for a, b in byte_spans:
        a = max(0, min(byte_len, int(a)))
        b = max(0, min(byte_len, int(b)))
        out.append((byte_to_char[a], byte_to_char[b]))
    return out


class FakeScorer:
    def score(self, text: str, expected: int, ds4_byte_offsets=None):
        # Test-only deterministic shape: later tokens score slightly higher.
        del text, ds4_byte_offsets
        if expected <= 0:
            return [], (0.0, 0.0, 0.0)
        return [i / max(1, expected - 1) for i in range(expected)], (0.0, 0.0, 0.0)


class MlxScorer:
    def __init__(self, args):
        try:
            import mlx.core as mx  # noqa: PLC0415
            import mlx_lm.spec_prefill as spec_prefill  # noqa: PLC0415
            from mlx_lm import load  # noqa: PLC0415
            from mlx_lm.spec_prefill import compute_keep_indices  # noqa: PLC0415
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "live drafter helper requires installed Python packages: "
                "mlx, mlx-lm, and transformers. Install them in the "
                "--spec-prefill-drafter-python environment."
            ) from exc

        self.mx = mx
        self.spec_prefill = spec_prefill
        self.compute_keep_indices = compute_keep_indices
        self.n_lookahead = args.n_lookahead
        self.pool_kernel = args.pool_kernel
        self.prefill_step_size = args.prefill_step_size
        self.keep_fraction = args.keep_fraction
        self.block_size = args.block_size
        self.sink_size = args.sink_size
        self.tail_keep = args.tail_keep

        print(f"ds4-live-drafter: load scorer tokenizer {args.scorer_model}", file=sys.stderr, flush=True)
        self.scorer_tk = AutoTokenizer.from_pretrained(args.scorer_model, trust_remote_code=True)
        print(f"ds4-live-drafter: load dsv4 tokenizer {args.dsv4_tokenizer}", file=sys.stderr, flush=True)
        self.dsv4_tk = AutoTokenizer.from_pretrained(args.dsv4_tokenizer, trust_remote_code=True)
        print(f"ds4-live-drafter: load MLX scorer model {args.scorer_model}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        self.model, _ = load(args.scorer_model)
        mx.eval(self.model.parameters())
        print(f"ds4-live-drafter: ready in {time.perf_counter() - t0:.2f}s", file=sys.stderr, flush=True)

    def score(self, text: str, expected: int, ds4_byte_offsets=None):
        t0 = time.perf_counter()
        scorer_enc = self.scorer_tk(text, return_offsets_mapping=True, add_special_tokens=False)
        scorer_ids = scorer_enc["input_ids"]
        scorer_offsets = list(scorer_enc["offset_mapping"])
        if os.environ.get("DS4_DRAFTER_DEBUG_TOKENS") == "1":
            if os.environ.get("DS4_DRAFTER_DEBUG_TOKEN_HASHES") == "1":
                block = 256
                hashes = []
                for pos in range(0, len(scorer_ids), block):
                    h = 0x14650FB0739D0383
                    for tid in scorer_ids[pos : pos + block]:
                        v = int(tid) & 0xFFFFFFFF
                        for k in range(4):
                            h ^= (v >> (k * 8)) & 0xFF
                            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                    hashes.append(f"{h:016x}")
                print(
                    "PY_TOKEN_HASHES",
                    len(scorer_ids),
                    block,
                    " ".join(hashes),
                    file=sys.stderr,
                    flush=True,
                )
            start, count = 0, 64
            window = os.environ.get("DS4_DRAFTER_DEBUG_TOKEN_WINDOW")
            if window:
                try:
                    start_s, count_s = window.split(":", 1)
                    start, count = max(0, int(start_s)), max(0, int(count_s))
                except ValueError:
                    start, count = 0, 64
            end = min(len(scorer_ids), start + count)
            start = min(start, len(scorer_ids))
            print(
                "PY_TOKENS",
                len(scorer_ids),
                start,
                end,
                " ".join(
                    f"{int(t)}:{int(a)}:{int(b)}"
                    for t, (a, b) in zip(scorer_ids[start:end], scorer_offsets[start:end])
                ),
                file=sys.stderr,
                flush=True,
            )
        if ds4_byte_offsets is not None:
            dsv4_offsets = byte_spans_to_char_spans(text, ds4_byte_offsets)
        else:
            dsv4_enc = self.dsv4_tk(text, return_offsets_mapping=True, add_special_tokens=False)
            dsv4_offsets = list(dsv4_enc["offset_mapping"])
        t1 = time.perf_counter()

        prompt = self.mx.array(scorer_ids, dtype=self.mx.uint32)
        keep = self.compute_keep_indices(
            self.model,
            prompt,
            lookahead_steps=self.n_lookahead,
            pool_kernel=self.pool_kernel,
            block_size=self.block_size,
            keep_fraction=self.keep_fraction,
            prefill_step_size=self.prefill_step_size,
            sink_size=self.sink_size,
            tail_keep=self.tail_keep,
            aggregation="pool_then_max",
        )
        self.mx.eval(keep)
        scorer_scores = [0.0] * len(scorer_ids)
        keep_list = [int(i) for i in keep.tolist()]
        for i in keep_list:
            if 0 <= int(i) < len(scorer_scores):
                scorer_scores[int(i)] = 1.0
        if os.environ.get("DS4_DRAFTER_DEBUG_KEEP") == "1":
            print(
                "PY_KEEP",
                len(scorer_ids),
                " ".join(str(i) for i in keep_list),
                file=sys.stderr,
                flush=True,
            )
        if os.environ.get("DS4_DRAFTER_DEBUG_BLOCKS") == "1":
            self._debug_print_block_scores(prompt)
        t2 = time.perf_counter()

        dsv4_scores = normalize(realign_to_dsv4(scorer_scores, scorer_offsets, dsv4_offsets))
        if len(dsv4_scores) != expected:
            raise ValueError(
                f"aligned scorer produced {len(dsv4_scores)} tokens, expected DS4 transcript has {expected}"
            )
        t3 = time.perf_counter()
        return dsv4_scores, ((t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0)

    def _debug_print_block_scores(self, prompt):
        sp = self.spec_prefill
        mx = self.mx
        spec_cache = sp.cache_module.make_prompt_cache(self.model)
        m = int(prompt.size)
        last_logits = sp._chunked_prefill(self.model, prompt, spec_cache, self.prefill_step_size)
        captured, handles = sp._install_query_capture(self.model)
        try:
            y = mx.argmax(last_logits[:, -1, :], axis=-1).astype(prompt.dtype)
            mx.eval(y)
            if os.environ.get("DS4_DRAFTER_DEBUG_ARGMAX") == "1":
                print("PY_BLOCK_ARGMAX", int(y.item()), end="", file=sys.stderr, flush=True)
            for _ in range(self.n_lookahead):
                sp._reset_capture_step(handles)
                logits = self.model(y[None], cache=spec_cache)
                y = mx.argmax(logits[:, -1, :], axis=-1).astype(prompt.dtype)
                mx.eval(y)
                if os.environ.get("DS4_DRAFTER_DEBUG_ARGMAX") == "1":
                    print(" " + str(int(y.item())), end="", file=sys.stderr, flush=True)
        finally:
            sp._remove_query_capture(handles)
        if os.environ.get("DS4_DRAFTER_DEBUG_ARGMAX") == "1":
            print(file=sys.stderr, flush=True)
        layer_keys = sp._speculator_keys_for_prompt(spec_cache, m)
        per_layer_probs = []
        for slot, keys in zip(captured, layer_keys):
            if not slot or keys is None:
                continue
            q_layer = mx.concatenate(slot, axis=2)
            per_layer_probs.append(sp._layer_softmax_probs(q_layer, keys, q_layer.shape[-1]))
        importance = sp._aggregate_pool_then_max(per_layer_probs, self.pool_kernel)
        mx.eval(importance)
        vals = [float(x) for x in importance.tolist()]
        n_blocks = (m + self.block_size - 1) // self.block_size
        block_scores = []
        for b in range(n_blocks):
            start = b * self.block_size
            block = vals[start : min(start + self.block_size, m)]
            if len(block) < self.block_size:
                block = block + [-1.0e30] * (self.block_size - len(block))
            block_scores.append((b, sum(block) / self.block_size))
        k = max(1, int(math.ceil(self.keep_fraction * n_blocks)))
        k = min(k, n_blocks)
        block_scores.sort(key=lambda x: (-x[1], x[0]))
        print(
            "PY_BLOCKS",
            m,
            self.block_size,
            n_blocks,
            k,
            " ".join(f"{b}:{s:.9g}" for b, s in block_scores),
            file=sys.stderr,
            flush=True,
        )


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
    ap.add_argument("--dsv4-tokenizer", default="./gguf/dsv4-tokenizer")
    ap.add_argument("--n-lookahead", type=int, default=4)
    ap.add_argument("--pool-kernel", type=int, default=13)
    ap.add_argument("--prefill-step-size", type=int, default=2048)
    ap.add_argument("--keep-fraction", type=float, default=0.3)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--sink-size", type=int, default=16)
    ap.add_argument("--tail-keep", type=int, default=256)
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
            if op not in {"SCORE", "SCORE2"}:
                raise ValueError(f"unknown op {op}")
            expected = int(expected_s)
            nbytes = int(nbytes_s)
            payload = stdin.read(nbytes)
            if len(payload) != nbytes:
                raise ValueError(f"short prompt payload {len(payload)}/{nbytes}")
            text = payload.decode("utf-8")
            ds4_offsets = None
            if op == "SCORE2":
                raw_spans = stdin.read(expected * 8)
                if len(raw_spans) != expected * 8:
                    raise ValueError(f"short span payload {len(raw_spans)}/{expected * 8}")
                vals = struct.unpack("<" + "II" * expected, raw_spans)
                ds4_offsets = list(zip(vals[0::2], vals[1::2]))
            scores, timings = scorer.score(text, expected, ds4_offsets)
            send_scores(scores, timings)
        except Exception as exc:  # keep the resident helper alive for debuggability
            send_error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
