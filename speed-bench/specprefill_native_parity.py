#!/usr/bin/env python3
"""Compare native in-process SpecPrefill drafter scores with the Python/MLX helper.

This is a development parity harness, not a user-facing drafter binary. It
builds a temporary executable that includes ds4_drafter.c so static native
helpers can be exercised directly, then sends the same SCORE2 request to the
resident Python helper and compares the aligned DS4-token score vectors.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_TEXT = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
    "nu xi omicron pi rho sigma tau"
)


HARNESS_C = r'''
#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ds4_drafter.c"

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e;
    (void)token;
    if (len) *len = 0;
    char *s = malloc(1);
    if (s) s[0] = '\0';
    return s;
}

static int read_file(const char *path, char **out, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return -1;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return -1;
    }
    char *buf = malloc((size_t)n + 1u);
    if (!buf) {
        fclose(fp);
        return -1;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    if (got != (size_t)n) {
        free(buf);
        return -1;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = (size_t)n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 11) {
        fprintf(stderr, "usage: %s MODEL TEXT_FILE N_SPANS LOOKAHEAD POOL KEEP SINK TAIL CHUNK WARMUPS\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *text_path = argv[2];
    int n_spans = atoi(argv[3]);
    int lookahead = atoi(argv[4]);
    int pool = atoi(argv[5]);
    float keep = (float)atof(argv[6]);
    int sink = atoi(argv[7]);
    int tail = atoi(argv[8]);
    int chunk = atoi(argv[9]);
    int warmups = atoi(argv[10]);
    if (warmups < 0) warmups = 0;
    char *text = NULL;
    size_t text_len = 0;
    if (read_file(text_path, &text, &text_len) != 0) {
        fprintf(stderr, "failed to read text file\n");
        return 3;
    }
    uint32_t *spans = calloc((size_t)n_spans * 2u, sizeof(spans[0]));
    if (!spans) {
        free(text);
        return 4;
    }
    for (int i = 0; i < n_spans; i++) {
        spans[(size_t)i * 2u + 0u] = (uint32_t)((size_t)i * text_len / (size_t)n_spans);
        spans[(size_t)i * 2u + 1u] = (uint32_t)((size_t)(i + 1) * text_len / (size_t)n_spans);
    }

    ds4_drafter d;
    ds4_drafter_init(&d);
    ds4_drafter_options opt = {
        .backend = DS4_DRAFTER_BACKEND_NATIVE,
        .model = model,
        .score_lookahead = lookahead,
        .score_pool_kernel = pool,
        .keep_pct = keep,
        .sink = sink,
        .tail = tail,
        .chunk = chunk,
        .process_name = "native-parity",
    };
    char err[1024] = {0};
    if (ds4_drafter_start(&d, &opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "start failed: %s\n", err);
        free(spans);
        free(text);
        return 5;
    }
    for (int w = 0; w < warmups; w++) {
        float *warm_scores = NULL;
        ds4_drafter_score_stats warm_stats = {0};
        if (native_drafter_score_text(&d, &opt, text, text_len, spans, n_spans,
                                      &warm_scores, &warm_stats, err, sizeof(err)) != 0) {
            fprintf(stderr, "warmup failed: %s\n", err);
            ds4_drafter_stop(&d);
            free(spans);
            free(text);
            return 6;
        }
        free(warm_scores);
    }
    float *scores = NULL;
    ds4_drafter_score_stats stats = {0};
    if (native_drafter_score_text(&d, &opt, text, text_len, spans, n_spans,
                                  &scores, &stats, err, sizeof(err)) != 0) {
        fprintf(stderr, "score failed: %s\n", err);
        ds4_drafter_stop(&d);
        free(spans);
        free(text);
        return 6;
    }
    printf("NATIVE_STATS total=%.3f tokenize=%.3f score=%.3f align=%.3f\n",
           stats.total_ms, stats.tokenize_ms, stats.score_ms, stats.align_ms);
    printf("NATIVE_SCORES");
    for (int i = 0; i < n_spans; i++) printf(" %.9g", scores[i]);
    printf("\n");
    const char *debug_tokens = getenv("DS4_DRAFTER_DEBUG_TOKENS");
    if (debug_tokens && strcmp(debug_tokens, "1") == 0) {
        int token_window_start = 0;
        int token_window_count = 64;
        const char *token_window = getenv("DS4_DRAFTER_DEBUG_TOKEN_WINDOW");
        if (token_window) {
            sscanf(token_window, "%d:%d", &token_window_start, &token_window_count);
            if (token_window_start < 0) token_window_start = 0;
            if (token_window_count < 0) token_window_count = 0;
        }
        ds4_drafter_native_model *nm = (ds4_drafter_native_model *)d.native;
        native_token_vec q_tokens = {0};
        if (nm &&
            native_tokenizer_encode_text(&nm->tokenizer, text, text_len,
                                         &q_tokens, err, sizeof(err)) == 0) {
            const char *debug_token_hashes = getenv("DS4_DRAFTER_DEBUG_TOKEN_HASHES");
            if (debug_token_hashes && strcmp(debug_token_hashes, "1") == 0) {
                const uint64_t fnv_offset = 1469598103934665603ull;
                const uint64_t fnv_prime = 1099511628211ull;
                const int hash_block = 256;
                int n_hash_blocks = (q_tokens.len + hash_block - 1) / hash_block;
                fprintf(stderr, "NATIVE_TOKEN_HASHES %d %d", q_tokens.len, hash_block);
                for (int b = 0; b < n_hash_blocks; b++) {
                    uint64_t h = fnv_offset;
                    int start = b * hash_block;
                    int end = start + hash_block;
                    if (end > q_tokens.len) end = q_tokens.len;
                    for (int i = start; i < end; i++) {
                        uint32_t id = (uint32_t)q_tokens.ids[i];
                        for (int k = 0; k < 4; k++) {
                            h ^= (uint64_t)((id >> (k * 8)) & 0xffu);
                            h *= fnv_prime;
                        }
                    }
                    fprintf(stderr, " %016llx", (unsigned long long)h);
                }
                fprintf(stderr, "\n");
            }
            int token_window_end = token_window_start + token_window_count;
            if (token_window_start > q_tokens.len) token_window_start = q_tokens.len;
            if (token_window_end > q_tokens.len) token_window_end = q_tokens.len;
            fprintf(stderr, "NATIVE_TOKENS %d %d %d", q_tokens.len,
                    token_window_start, token_window_end);
            for (int i = token_window_start; i < token_window_end; i++) {
                fprintf(stderr, " %d:%u:%u", q_tokens.ids[i],
                        q_tokens.offsets[(size_t)i * 2u + 0u],
                        q_tokens.offsets[(size_t)i * 2u + 1u]);
            }
            fprintf(stderr, "\n");
        } else {
            fprintf(stderr, "NATIVE_TOKENS_ERR %s\n", err);
        }
        native_token_vec_free(&q_tokens);
    }
    const char *debug_keep = getenv("DS4_DRAFTER_DEBUG_KEEP");
    if (debug_keep && strcmp(debug_keep, "1") == 0) {
        ds4_drafter_native_model *nm = (ds4_drafter_native_model *)d.native;
        native_token_vec q_tokens = {0};
        float *q_scores = NULL;
        if (nm &&
            native_tokenizer_encode_text(&nm->tokenizer, text, text_len,
                                         &q_tokens, err, sizeof(err)) == 0 &&
            native_score_prompt_tokens(nm, &opt, &q_tokens, &q_scores,
                                       err, sizeof(err)) == 0) {
            printf("NATIVE_KEEP %d", q_tokens.len);
            for (int i = 0; i < q_tokens.len; i++) {
                if (q_scores[i] > 0.5f) printf(" %d", i);
            }
            printf("\n");
        } else {
            printf("NATIVE_KEEP_ERR %s\n", err);
        }
        free(q_scores);
        native_token_vec_free(&q_tokens);
    }
    free(scores);
    ds4_drafter_stop(&d);
    free(spans);
    free(text);
    return 0;
}
'''


def build_native(repo: Path, tmp: Path, native_metal: bool) -> Path:
    src = tmp / "native_parity.c"
    exe = tmp / "native_parity"
    src.write_text(HARNESS_C)
    native_cpu_flag = "-mcpu=native" if sys.platform == "darwin" else "-march=native"
    cmd = [
        "cc",
        "-O3",
        "-ffast-math",
        "-g",
        native_cpu_flag,
        "-Wall",
        "-Wextra",
        "-std=c99",
        "-I.",
        str(src),
        "-o",
        str(exe),
        "-lm",
    ]
    if sys.platform == "darwin":
        cmd.insert(2, "-DACCELERATE_NEW_LAPACK")
        cmd.extend(["-framework", "Accelerate"])
        if native_metal:
            cmd.insert(2, "-DDS4_DRAFTER_HAS_METAL")
            cmd.extend([
                "-fobjc-arc",
                "ds4_drafter_metal.m",
                "-framework", "Foundation",
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
            ])
    elif native_metal:
        raise RuntimeError("--native-metal is only supported on Darwin/Metal builds")
    subprocess.run(cmd, cwd=repo, check=True)
    return exe


def parse_scores_line(prefix: str, text: str) -> list[float]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return [float(x) for x in line.split()[1:]]
    raise RuntimeError(f"missing {prefix} line:\n{text}")


def run_native(exe: Path, args, text_file: Path) -> tuple[list[float], str]:
    cmd = [
        str(exe),
        args.model,
        str(text_file),
        str(args.spans),
        str(args.lookahead),
        str(args.pool_kernel),
        str(args.keep_fraction),
        str(args.sink_size),
        str(args.tail_keep),
        str(args.block_size),
        str(args.native_warmups),
    ]
    env = os.environ.copy()
    if args.native_metal:
        env["DS4_DRAFTER_METAL"] = "1"
        env["DS4_DRAFTER_METAL_STRICT"] = "1"
    try:
        cp = subprocess.run(cmd, cwd=args.repo, env=env, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout, end="", file=sys.stderr)
        if e.stderr:
            print(e.stderr, end="", file=sys.stderr)
        raise
    return parse_scores_line("NATIVE_SCORES", cp.stdout), cp.stdout + cp.stderr


def run_python(args, text: bytes) -> tuple[list[float], str]:
    total_t0 = time.perf_counter()
    spans: list[int] = []
    for i in range(args.spans):
        spans.extend([i * len(text) // args.spans, (i + 1) * len(text) // args.spans])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.mlx_lm)
    cmd = [
        args.drafter_python,
        "-u",
        args.drafter_script,
        "--scorer-model",
        args.model,
        "--dsv4-tokenizer",
        args.dsv4_tokenizer,
        "--n-lookahead",
        str(args.lookahead),
        "--pool-kernel",
        str(args.pool_kernel),
        "--keep-fraction",
        str(args.keep_fraction),
        "--block-size",
        str(args.block_size),
        "--sink-size",
        str(args.sink_size),
        "--tail-keep",
        str(args.tail_keep),
    ]
    p = subprocess.Popen(
        cmd,
        cwd=args.repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert p.stdin and p.stdout and p.stderr
    ready = p.stdout.readline().decode("utf-8", errors="replace").strip()
    if ready != "READY":
        raise RuntimeError(f"python helper did not become ready: {ready}")
    score_t0 = time.perf_counter()
    p.stdin.write(f"SCORE2 {args.spans} {len(text)}\n".encode("ascii"))
    p.stdin.write(text)
    p.stdin.write(struct.pack("<" + "II" * args.spans, *spans))
    p.stdin.flush()
    line = p.stdout.readline().decode("utf-8", errors="replace").strip()
    parts = line.split()
    if len(parts) < 2 or parts[0] != "OK":
        raise RuntimeError(f"python helper error: {line}")
    n = int(parts[1])
    raw = p.stdout.read(n * 4)
    score_ms = (time.perf_counter() - score_t0) * 1000.0
    if len(raw) != n * 4:
        raise RuntimeError(f"python helper truncated score payload {len(raw)}/{n * 4}")
    scores = list(struct.unpack("<" + "f" * n, raw))
    try:
        p.stdin.write(b"QUIT\n")
        p.stdin.flush()
    except BrokenPipeError:
        pass
    stderr = p.stderr.read().decode("utf-8", errors="replace")
    p.wait(timeout=5)
    total_ms = (time.perf_counter() - total_t0) * 1000.0
    return scores, f"PYTHON_STATS total={total_ms:.3f} score={score_ms:.3f}\n{line}\n{stderr}"


def make_text(args) -> bytes:
    if args.text_file:
        return Path(args.text_file).read_bytes()
    unit = args.text or DEFAULT_TEXT
    text = "\n".join(unit for _ in range(args.repeat))
    return text.encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model", default="/Users/Shared/models/qwen3.5-0.8b-mlx-4bit")
    ap.add_argument("--dsv4-tokenizer", default="/Users/Shared/models/ds4-gguf/dsv4-tokenizer")
    ap.add_argument("--mlx-lm", type=Path, default=Path("/Users/carl/projects/mlx-lm"))
    ap.add_argument("--drafter-python", default="/Users/carl/projects/anemll-project/env-anemll/bin/python3")
    ap.add_argument("--drafter-script", default="speed-bench/ds4_live_drafter.py")
    ap.add_argument("--text")
    ap.add_argument("--text-file")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--spans", type=int, default=19)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--pool-kernel", type=int, default=3)
    ap.add_argument("--keep-fraction", type=float, default=0.3)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--sink-size", type=int, default=0)
    ap.add_argument("--tail-keep", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--native-only", action="store_true",
                    help="Run only the native scorer smoke without Python/MLX parity.")
    ap.add_argument("--native-metal", action="store_true",
                    help="Exercise the native Metal Qwen drafter path instead of CPU matvecs.")
    ap.add_argument("--native-warmups", type=int, default=0,
                    help="Run and discard this many native scores before the measured native score.")
    args = ap.parse_args()

    text = make_text(args)
    with tempfile.TemporaryDirectory(prefix="ds4_native_parity_") as td:
        tmp = Path(td)
        text_file = tmp / "prompt.txt"
        text_file.write_bytes(text)
        exe = build_native(args.repo, tmp, args.native_metal)
        native_scores, native_log = run_native(exe, args, text_file)
    if args.native_only:
        print(native_log, end="" if native_log.endswith("\n") else "\n")
        print("DS4_NATIVE_ONLY_OK")
        return 0
    python_scores, python_log = run_python(args, text)

    if len(native_scores) != len(python_scores):
        print(native_log)
        print(python_log)
        raise SystemExit(f"score length mismatch native={len(native_scores)} python={len(python_scores)}")
    diffs = [abs(a - b) for a, b in zip(native_scores, python_scores)]
    max_diff = max(diffs, default=0.0)
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    print(native_log, end="" if native_log.endswith("\n") else "\n")
    if python_log:
        first_python_log = python_log.splitlines()[0]
        if first_python_log.startswith("PYTHON_STATS"):
            print(first_python_log)
    print("PYTHON_SCORES", " ".join(f"{x:.9g}" for x in python_scores))
    print(f"PARITY max_abs={max_diff:.9g} mean_abs={mean_diff:.9g} n={len(diffs)}")
    if os.environ.get("DS4_DRAFTER_DEBUG_KEEP") == "1" and python_log:
        print(python_log, file=sys.stderr)
    if max_diff > args.tolerance:
        if os.environ.get("DS4_DRAFTER_DEBUG_KEEP") != "1":
            print(python_log, file=sys.stderr)
        return 1
    print("DS4_NATIVE_PARITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
