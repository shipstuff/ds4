#!/usr/bin/env python3
"""Drive ./ds4 through a scripted multi-turn chat, capture per-turn
SpecPrefill / timing metrics, write a CSV, and plot the comparison.

Designed for the live-drafter SpecPrefill path. Runs the same conversation
under baseline and drafter modes and surfaces:

  - per-turn prefill t/s and generation t/s (parsed from ds4's own
    DS4_LOG_TIMING line)
  - compression ratio per turn (parsed from the `spec-prefill:` line)
  - per-turn wall time (subprocess measurement)
  - optional CPU vs Metal max-abs-diff per turn (DS4_SCORE_VALIDATE=1)

Output:
  $OUT_DIR/metrics.csv      raw rows: mode, turn, ...
  $OUT_DIR/<mode>.stderr    full ds4 stderr per mode (for debugging)
  $OUT_DIR/plot_*.png       matplotlib comparison plots
                            (skip with --no-plot or if matplotlib is
                            not installed)

Quick start on a 96 GB+ Mac:

  python speed-bench/specprefill_chat_loop.py \\
      --model ./ds4flash.gguf \\
      --modes baseline drafter \\
      --drafter-model ./gguf/qwen3.5-0.8b-mlx-4bit \\
      --out-dir /tmp/ds4_chatloop

Use --validate only when checking scorer parity; it enables DS4_SCORE_VALIDATE
and slows the run.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import select
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

# linenoise (ds4's REPL line editor) chokes on huge single-line inputs
# over a piped stdin.  For any turn longer than this we write it to a
# tempfile and trigger ds4's `/read FILE` REPL command instead, which
# bypasses linenoise's line buffer.
LARGE_TURN_BYTES = 4 * 1024

# ds4's own log lines.  These come from ds4_cli.c (DS4_LOG_PREFILL,
# DS4_LOG_TIMING) and from ds4.c (DS4_SCORE_VALIDATE path).  Anchored on
# the parts that don't change run-to-run.
RE_SPECPREFILL = re.compile(
    r"spec-prefill:\s*(?:(?:fresh|cold|recompress)\s+)?prompt\s*(\d+)\s*->\s*(\d+)\s*tokens\s*"
    r"\(keep=([\d.]+)\s*(?:sink=(\d+)\s*)?tail=(\d+)\s*chunk=(\d+)\s*scores=(\S+)\)"
)
RE_TIMING = re.compile(
    r"prefill:\s*([\d.]+)\s*t/s,\s*generation:\s*([\d.]+)\s*t/s,\s*"
    r"ctx:\s*(\d+)/(\d+)\s*remaining=(\d+)\s*transcript=(\d+)"
)
RE_TIMING_LEGACY = re.compile(
    r"prefill:\s*([\d.]+)\s*t/s,\s*generation:\s*([\d.]+)\s*t/s"
)
RE_TTFT = re.compile(
    r"ttft:\s*(?P<ttft_ms>[\d.]+)\s*ms\s*"
    r"\(drafter=(?P<drafter_ms>[\d.]+)\s*ms\s*compress=(?P<compress_ms>[\d.]+)\s*ms\s*"
    r"prefill=(?P<target_prefill_ms>[\d.]+)\s*ms\s*first_decode=(?P<first_decode_ms>[\d.]+)\s*ms\)\s*"
    r"tokens:\s*canonical=(?P<canonical_tokens>\d+)\s*sync=(?P<sync_tokens>\d+)\s*"
    r"suffix=(?P<suffix_tokens>\d+)\s*generated_first=1"
    r"(?:\s*rates:\s*effective_prompt=(?P<effective_prompt_tps>[\d.]+)\s*t/s\s*"
    r"target_prefill=(?P<target_prefill_tps>[\d.]+)\s*t/s\s*"
    r"drafter=(?P<drafter_tps>[\d.]+)\s*t/s"
    r"(?:\s*generation=(?P<gen_tps>[\d.]+)\s*t/s\s*"
    r"ctx=(?P<ctx_session_tokens>\d+)/(?P<ctx_limit>\d+)\s*"
    r"remaining=(?P<ctx_remaining>\d+)\s*transcript=(?P<transcript_tokens>\d+))?)?"
)
RE_VALIDATE = re.compile(
    r"spec-prefill validate:.*?max\|m-c\|=([\d.]+).*?mean\|m-c\|=([\d.]+)\s*rms=([\d.]+)"
)


@dataclass
class TurnMetrics:
    mode: str
    turn: int
    depth: int | None = None
    prompt_tokens: int | None = None
    compressed_tokens: int | None = None
    scores_source: str | None = None
    prefill_tps: float | None = None
    gen_tps: float | None = None
    ttft_ms: float | None = None
    drafter_ms: float | None = None
    compress_ms: float | None = None
    target_prefill_ms: float | None = None
    first_decode_ms: float | None = None
    canonical_tokens: int | None = None
    sync_tokens: int | None = None
    suffix_tokens: int | None = None
    effective_prompt_tps: float | None = None
    target_prefill_tps: float | None = None
    drafter_tps: float | None = None
    ctx_session_tokens: int | None = None
    ctx_limit: int | None = None
    ctx_remaining: int | None = None
    transcript_tokens: int | None = None
    generated_tokens: int | None = None
    validate_max_abs: float | None = None
    validate_rms: float | None = None
    wall_s: float | None = None


@dataclass
class ModeRun:
    label: str
    args_extra: list[str] = field(default_factory=list)


SECRET_PHRASE = "Obsidian Falcon"
VAULT_CODE = "4827193"


CITIES = [
    "Aldermouth", "Brindlecross", "Caldwater", "Dunmore", "Ellingate",
    "Fernwick", "Glasholm", "Harrowby", "Ivescombe", "Jarrowdale",
    "Kelmsford", "Marlowe", "Netherby", "Orrindale",
]
GOODS = [
    "copper ingots", "dried figs", "sailcloth", "oak staves",
    "glass beads", "iron nails", "wool bales", "salt cod",
    "tin sheets", "amber resin", "linen thread", "clay tiles",
    "beeswax", "hemp rope", "cured leather",
]
NAMES = [
    "Aria", "Boren", "Cass", "Delia", "Emrik", "Fenna", "Goran",
    "Hessa", "Ivo", "Juna", "Kael", "Lira", "Moss", "Neve", "Oren",
    "Pell", "Quist", "Rhea",
]


def build_modes(args) -> list[ModeRun]:
    modes: list[ModeRun] = []
    if "baseline" in args.modes:
        modes.append(ModeRun("baseline", []))
    if "drafter" in args.modes:
        if not args.drafter_model:
            raise ValueError("--drafter-model is required for --modes drafter")
        extra = [
            f"--spec-prefill={args.keep_pct}",
            "--spec-prefill-sink", str(args.sink),
            "--spec-prefill-tail", str(args.tail),
            "--spec-prefill-chunk", str(args.chunk),
            "--spec-prefill-drafter-model", args.drafter_model,
            "--spec-prefill-score-lookahead", str(args.drafter_score_lookahead),
            "--spec-prefill-score-pool-kernel", str(args.drafter_score_pool_kernel),
        ]
        if args.drafter_backend == "native":
            extra.append("--spec-prefill-drafter-native")
        else:
            extra += [
                "--spec-prefill-drafter-python", args.drafter_python,
                "--spec-prefill-drafter-script", args.drafter_script,
                "--spec-prefill-drafter-tokenizer", args.drafter_tokenizer,
            ]
        extra += ["--spec-prefill-cache", args.spec_prefill_cache]
        modes.append(ModeRun("drafter", extra))
    return modes


def needle_filler_line(rng: random.Random, i: int) -> str:
    city = rng.choice(CITIES)
    good = rng.choice(GOODS)
    name = rng.choice(NAMES)
    qty = rng.randint(12, 9800)
    price = rng.randint(3, 290)
    day = rng.randint(1, 28)
    month = rng.choice(["Janus", "Feber", "Marlin", "Aprion", "Mayes", "Junet"])
    return (
        f"Ledger entry {i:04d}: on the {day}th of {month}, {name} of "
        f"{city} shipped {qty} units of {good} at {price} marks each; "
        "the consignment cleared customs without dispute and was logged "
        "for quarterly review."
    )


def build_needle_prompt(depth_pct: int, target_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    lines: list[str] = []
    total = 0
    i = 1
    while total < target_chars:
        line = needle_filler_line(rng, i)
        lines.append(line)
        total += len(line) + 1
        i += 1

    needle = (
        f"IMPORTANT FACT TO REMEMBER: the secret pass phrase is "
        f"\"{SECRET_PHRASE}\" and the vault access code is {VAULT_CODE}. "
        "Keep this in mind for the question at the end."
    )
    pos = int(len(lines) * depth_pct / 100.0)
    pos = max(0, min(len(lines), pos))
    lines.insert(pos, needle)
    intro = (
        "You are reviewing a long shipping ledger. Read the entire document "
        "carefully; there is one important fact embedded in it that you will "
        "be asked about later.\n\n"
    )
    return intro + "\n".join(lines)


def build_needle_turns(depth_pct: int, target_chars: int, seed: int) -> list[str]:
    return [
        build_needle_prompt(depth_pct, target_chars, seed),
        (
            "Earlier in this document a secret pass phrase and a vault access "
            "code were stated. What is the secret pass phrase, and what is "
            "the 7-digit vault access code? Answer with exactly the phrase "
            "in quotes followed by the number."
        ),
    ]


def classify_coherence(text: str) -> tuple[str, dict[str, float | int]]:
    words = text.strip().split()
    n = len(words)
    if n < 3:
        return "empty", {"words": n}
    uniq_ratio = len({w.lower() for w in words}) / n
    max_consec = 1
    for plen in range(1, 7):
        i = 0
        while i + 2 * plen <= n:
            span = words[i : i + plen]
            reps = 1
            j = i + plen
            while j + plen <= n and words[j : j + plen] == span:
                reps += 1
                j += plen
            max_consec = max(max_consec, reps)
            i += 1
    grams = [" ".join(words[i : i + 3]).lower() for i in range(n - 2)]
    top_gram = Counter(grams).most_common(1)[0][1] if grams else 0
    gram_share = top_gram / max(1, len(grams))
    metrics = {
        "words": n,
        "uniq_ratio": round(uniq_ratio, 3),
        "max_consec_repeat": max_consec,
        "top_3gram_share": round(gram_share, 3),
    }
    if max_consec >= 5 or (n >= 12 and top_gram >= 4 and gram_share > 0.20):
        return "repetition-loop", metrics
    if uniq_ratio < 0.30:
        return "low-diversity", metrics
    return "coherent", metrics


def grade_needle_output(text: str) -> dict[str, object]:
    has_phrase = SECRET_PHRASE.lower() in text.lower()
    has_code = VAULT_CODE in text
    seven_digits = re.findall(r"\b\d{7}\b", text)[:5]
    coherence, metrics = classify_coherence(text)
    return {
        "recall_exact": has_phrase and has_code,
        "has_phrase": has_phrase,
        "has_code": has_code,
        "coherence": coherence,
        "pass": has_phrase and has_code and coherence == "coherent",
        "seven_digits": " ".join(seven_digits),
        **metrics,
    }


def load_turns(args) -> list[str]:
    """Build the multi-turn script.  Turn 1 is the long prompt
    (multi-line content is fine -- the runner uses ds4's /read REPL
    command for large turns); subsequent turns are short follow-ups
    that exercise the model's memory of the compressed history."""
    if args.turns_file:
        # One turn per non-empty line.
        return [t.rstrip("\n") for t in Path(args.turns_file).read_text().splitlines() if t.strip()]
    first = Path(args.prompt_file).read_text().rstrip()
    return [first] + list(args.follow_ups)


def feed_repl(
    args,
    mode: ModeRun,
    turns: list[str],
    stderr_path: Path,
    stdout_path: Path,
    *,
    depth: int | None = None,
) -> list[TurnMetrics]:
    """Run ds4 once with the given mode flags, pipe each turn through stdin,
    parse stderr live to attribute log lines to turn indices."""
    base_args = [
        args.ds4,
        "-m", args.model,
        "--ctx", str(args.ctx),
        "--nothink",
        "-n", str(args.n_predict),
        "--temp", str(args.temp),
    ]
    if args.backend:
        base_args += ["--backend", args.backend]
    if args.warm_weights:
        base_args += ["--warm-weights"]
    if args.dist_coordinator:
        base_args += [
            "--role", "coordinator",
            "--layers", args.dist_layers,
            "--listen", args.dist_listen_host, str(args.dist_listen_port),
        ]
        if args.dist_prefill_chunk:
            base_args += ["--dist-prefill-chunk", str(args.dist_prefill_chunk)]
        if args.dist_prefill_window:
            base_args += ["--dist-prefill-window", str(args.dist_prefill_window)]
        if args.dist_activation_bits:
            base_args += ["--dist-activation-bits", str(args.dist_activation_bits)]
        if args.dist_debug:
            base_args += ["--debug"]
    cmd = base_args + mode.args_extra
    env = os.environ.copy()
    if args.validate:
        env["DS4_SCORE_VALIDATE"] = "1"
    if args.dist_socket_timeout_sec:
        env["DS4_DIST_SOCKET_TIMEOUT_SEC"] = str(args.dist_socket_timeout_sec)

    print(f"\n=== {mode.label} ===", file=sys.stderr)
    print("  " + " ".join(cmd), file=sys.stderr)

    # Long-turn tempfiles live until the mode finishes so /read can
    # finish slurping them.  Cleaned up in the finally block.
    tmp_paths: list[Path] = []

    turns_metrics: list[TurnMetrics] = []
    with stdout_path.open("w") as stdout_fp:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=stdout_fp,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None and proc.stderr is not None
        stderr_fd = proc.stderr.fileno()

        def stdin_write(s: str) -> bool:
            """Write to ds4's stdin.  Returns False if the child has exited."""
            if proc.poll() is not None:
                return False
            try:
                proc.stdin.write(s)
                proc.stdin.flush()
                return True
            except BrokenPipeError:
                return False

        with stderr_path.open("w") as stderr_fp:
            def read_stderr_line(deadline: float) -> str:
                while True:
                    if proc.poll() is not None:
                        return proc.stderr.readline()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        proc.kill()
                        raise TimeoutError(
                            f"timed out waiting for ds4 timing line after {args.timeout}s"
                        )
                    ready, _, _ = select.select([stderr_fd], [], [], min(1.0, remaining))
                    if ready:
                        return proc.stderr.readline()

            def drain_stderr_available(grace_s: float = 0.25) -> None:
                """Drain late non-boundary lines from the previous turn."""
                end = time.monotonic() + grace_s
                while time.monotonic() < end:
                    ready, _, _ = select.select([stderr_fd], [], [], max(0.0, end - time.monotonic()))
                    if not ready:
                        return
                    line = proc.stderr.readline()
                    if not line:
                        return
                    stderr_fp.write(line)
                    stderr_fp.flush()

            for i, turn in enumerate(turns):
                if i > 0:
                    drain_stderr_available()
                if args.chat_cache == "cold-full-prompt" and i > 0:
                    # Match mini-01's `--mode cold --chat-loop`: keep the
                    # model process resident, but reset the live KV session
                    # before each follow-up. The DS4 transcript is preserved,
                    # so the next turn prefills the full accumulated prompt.
                    if not stdin_write(f"/ctx {args.ctx}\n"):
                        exit_code = proc.returncode if proc.poll() is not None else "<still running, pipe closed>"
                        print(
                            f"  [warn] ds4 stdin closed before /ctx reset for turn {i} "
                            f"(exit={exit_code}); stopping after captured turns",
                            file=sys.stderr,
                        )
                        break
                    drain_stderr_available(0.75)
                current = TurnMetrics(mode=mode.label, turn=i, depth=depth)
                t_started = time.perf_counter()
                deadline = time.monotonic() + args.timeout

                # Long turns go via /read FILE to bypass linenoise's line
                # buffer; short turns go inline.
                payload = turn + "\n"
                if len(payload.encode("utf-8")) > LARGE_TURN_BYTES:
                    fd, tmp = tempfile.mkstemp(
                        prefix=f"ds4_chatloop_{mode.label}_t{i}_",
                        suffix=".txt", dir=str(stderr_path.parent),
                    )
                    os.close(fd)
                    tmp_paths.append(Path(tmp))
                    Path(tmp).write_text(turn)
                    ok = stdin_write(f"/read {tmp}\n")
                else:
                    ok = stdin_write(payload)

                if not ok:
                    exit_code = proc.returncode if proc.poll() is not None else "<still running, pipe closed>"
                    print(
                        f"  [warn] ds4 stdin closed before turn {i} could be sent "
                        f"(exit={exit_code}); stopping after captured turns",
                        file=sys.stderr,
                    )
                    break

                # Block-read stderr until we see the timing line for this turn
                # or the process exits.
                while True:
                    try:
                        line = read_stderr_line(deadline)
                    except TimeoutError as exc:
                        print(f"  [warn] {exc}", file=sys.stderr)
                        break
                    if not line:
                        break
                    stderr_fp.write(line)
                    stderr_fp.flush()
                    m = RE_SPECPREFILL.search(line)
                    if m:
                        current.prompt_tokens = int(m.group(1))
                        current.compressed_tokens = int(m.group(2))
                        current.scores_source = m.group(7)
                    m = RE_VALIDATE.search(line)
                    if m:
                        current.validate_max_abs = float(m.group(1))
                        current.validate_rms = float(m.group(3))
                    m = RE_TTFT.search(line)
                    if m:
                        current.ttft_ms = float(m.group("ttft_ms"))
                        current.drafter_ms = float(m.group("drafter_ms"))
                        current.compress_ms = float(m.group("compress_ms"))
                        current.target_prefill_ms = float(m.group("target_prefill_ms"))
                        current.first_decode_ms = float(m.group("first_decode_ms"))
                        current.canonical_tokens = int(m.group("canonical_tokens"))
                        current.sync_tokens = int(m.group("sync_tokens"))
                        current.suffix_tokens = int(m.group("suffix_tokens"))
                        current.prompt_tokens = current.canonical_tokens
                        if m.group("effective_prompt_tps"):
                            current.effective_prompt_tps = float(m.group("effective_prompt_tps"))
                            current.target_prefill_tps = float(m.group("target_prefill_tps"))
                            current.drafter_tps = float(m.group("drafter_tps"))
                            if m.group("gen_tps"):
                                current.gen_tps = float(m.group("gen_tps"))
                                current.prefill_tps = current.target_prefill_tps
                                current.ctx_session_tokens = int(m.group("ctx_session_tokens"))
                                current.ctx_limit = int(m.group("ctx_limit"))
                                current.ctx_remaining = int(m.group("ctx_remaining"))
                                current.transcript_tokens = int(m.group("transcript_tokens"))
                                if current.sync_tokens is not None:
                                    current.generated_tokens = max(
                                        0,
                                        current.ctx_session_tokens - current.sync_tokens,
                                    )
                                current.wall_s = time.perf_counter() - t_started
                                turns_metrics.append(current)
                                break
                        else:
                            ttft_s = current.ttft_ms / 1000.0
                            target_s = current.target_prefill_ms / 1000.0
                            drafter_s = current.drafter_ms / 1000.0
                            current.effective_prompt_tps = (
                                current.canonical_tokens / ttft_s if ttft_s > 0 else None
                            )
                            current.target_prefill_tps = (
                                current.suffix_tokens / target_s if target_s > 0 else None
                            )
                            current.drafter_tps = (
                                current.canonical_tokens / drafter_s if drafter_s > 0 else None
                            )
                    m = RE_TIMING.search(line)
                    if m:
                        if current.ttft_ms is None:
                            continue
                        current.prefill_tps = float(m.group(1))
                        current.gen_tps = float(m.group(2))
                        current.ctx_session_tokens = int(m.group(3))
                        current.ctx_limit = int(m.group(4))
                        current.ctx_remaining = int(m.group(5))
                        current.transcript_tokens = int(m.group(6))
                        if current.sync_tokens is not None:
                            current.generated_tokens = max(
                                0,
                                current.ctx_session_tokens - current.sync_tokens,
                            )
                        current.wall_s = time.perf_counter() - t_started
                        turns_metrics.append(current)
                        break
                    m = RE_TIMING_LEGACY.search(line)
                    if m:
                        if current.ttft_ms is None:
                            continue
                        current.prefill_tps = float(m.group(1))
                        current.gen_tps = float(m.group(2))
                        current.wall_s = time.perf_counter() - t_started
                        turns_metrics.append(current)
                        break
                if proc.poll() is not None or (
                    current.prefill_tps is None and current.gen_tps is None
                ):
                    if current.prompt_tokens is not None or current.compressed_tokens is not None:
                        current.wall_s = time.perf_counter() - t_started
                        turns_metrics.append(current)
                    print(
                        f"  [warn] turn {i} did not produce a timing line; "
                        f"check {stderr_path}",
                        file=sys.stderr,
                    )
                    break

            try:
                stdin_write("/quit\n")
                proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                tail = proc.stderr.read()
                if tail:
                    stderr_fp.write(tail)
                    stderr_fp.flush()
            except Exception:
                pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        for p in tmp_paths:
            try:
                p.unlink()
            except OSError:
                pass

    return turns_metrics


def write_csv(metrics: list[TurnMetrics], out_path: Path) -> None:
    fields = [
        "mode", "depth", "turn",
        "prompt_tokens", "compressed_tokens", "scores_source",
        "prefill_tps", "gen_tps", "wall_s",
        "ttft_ms", "drafter_ms", "compress_ms", "target_prefill_ms", "first_decode_ms",
        "canonical_tokens", "sync_tokens", "suffix_tokens",
        "effective_prompt_tps", "target_prefill_tps", "drafter_tps",
        "ctx_session_tokens", "ctx_limit", "ctx_remaining", "transcript_tokens",
        "generated_tokens",
        "validate_max_abs", "validate_rms",
    ]
    with out_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(fields)
        for m in metrics:
            w.writerow([getattr(m, f) for f in fields])


def read_metrics_csv(path: Path) -> list[TurnMetrics]:
    int_fields = {
        "depth", "turn", "prompt_tokens", "compressed_tokens",
        "canonical_tokens", "sync_tokens", "suffix_tokens",
        "ctx_session_tokens", "ctx_limit", "ctx_remaining", "transcript_tokens",
        "generated_tokens",
    }
    float_fields = {
        "prefill_tps", "gen_tps", "wall_s", "ttft_ms", "drafter_ms",
        "compress_ms", "target_prefill_ms", "first_decode_ms",
        "effective_prompt_tps", "target_prefill_tps", "drafter_tps",
        "validate_max_abs", "validate_rms",
    }
    known_fields = {f.name for f in fields(TurnMetrics)}
    metrics: list[TurnMetrics] = []
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            kwargs = {}
            for key, raw_value in row.items():
                if key not in known_fields:
                    continue
                value = raw_value if raw_value != "" else None
                if value is not None and key in int_fields:
                    value = int(value)
                elif value is not None and key in float_fields:
                    value = float(value)
                kwargs[key] = value
            metrics.append(TurnMetrics(**kwargs))
    return metrics


def write_needle_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "mode", "depth", "recall_exact", "has_phrase", "has_code",
        "coherence", "pass", "seven_digits", "words", "uniq_ratio",
        "max_consec_repeat", "top_3gram_share", "stdout_path",
    ]
    with out_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field) for field in fields})


def maybe_plot(metrics: list[TurnMetrics], out_dir: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots "
              "(install with `pip install matplotlib` to enable)", file=sys.stderr)
        return False

    modes = sorted({m.mode for m in metrics})

    def x_context(m: TurnMetrics) -> int:
        return m.transcript_tokens or m.canonical_tokens or m.prompt_tokens or m.turn

    def plot_value(m: TurnMetrics, attr: str):
        is_full_sync = (
            m.suffix_tokens is not None and
            m.canonical_tokens is not None and
            m.suffix_tokens == m.canonical_tokens
        )
        is_recompress = m.compressed_tokens is not None
        if attr == "ttft_ms":
            if not (is_full_sync or is_recompress):
                return None
        if attr == "effective_prompt_tps":
            if not (is_full_sync or is_recompress):
                return None
        return getattr(m, attr)

    plots = [
        ("wall_s",            "Per-turn wall time (s)",       "lower is better"),
        ("effective_prompt_tps", "Effective prompt tok/s (TTFT throughput)", "full canonical tokens / TTFT"),
        ("ttft_ms",           "TTFT (ms)",                    "lower is better"),
        ("prefill_tps",       "Prefill throughput (t/s)",     "higher is better"),
        ("target_prefill_tps", "Target prefill throughput (t/s)", "actual synced suffix tokens / target prefill"),
        ("gen_tps",           "Generation throughput (t/s)",  "higher is better"),
        ("compressed_tokens", "Compressed prompt tokens",     "lower = more compression"),
    ]

    for attr, title, hint in plots:
        fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
        for mode in modes:
            xs, ys = [], []
            for m in sorted((m for m in metrics if m.mode == mode), key=x_context):
                y = plot_value(m, attr)
                if y is not None:
                    xs.append(x_context(m))
                    ys.append(y)
            if not ys:
                continue
            ax.plot(xs, ys, marker="o", label=mode)
        ax.set_xlabel("Total logical context tokens after turn")
        ax.set_ylabel(title)
        ax.set_title(f"{title} ({hint})")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        out_path = out_dir / f"plot_{attr}.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"wrote {out_path}", file=sys.stderr)

    # Parity plot (only if validate ran).
    if any(m.validate_max_abs is not None for m in metrics):
        fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
        for mode in modes:
            xs, ys = [], []
            for m in sorted((m for m in metrics if m.mode == mode), key=x_context):
                y = getattr(m, "validate_max_abs")
                if y is not None:
                    xs.append(x_context(m))
                    ys.append(y)
            if not ys:
                continue
            ax.plot(xs, ys, marker="o", label=mode)
        ax.set_xlabel("Total logical context tokens after turn")
        ax.set_ylabel("max | metal-cpu |")
        ax.set_title("CPU vs Metal scorer parity per turn (lower is better)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        out_path = out_dir / "plot_validate_max_abs.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"wrote {out_path}", file=sys.stderr)

    return True


def write_comparison_report(metrics: list[TurnMetrics], out_dir: Path, title: str) -> None:
    rows = [m for m in metrics if m.turn is not None]
    if not rows:
        return
    modes = sorted({m.mode for m in rows})
    turns = sorted({m.turn for m in rows})
    by_mode_turn = {(m.mode, m.turn): m for m in rows}

    def x_context(m: TurnMetrics) -> int:
        return m.transcript_tokens or m.canonical_tokens or m.prompt_tokens or m.turn

    def chart_value(m: TurnMetrics, attr: str):
        if attr == "ttft_ms":
            # Warm suffix-prefill timings are not full-context TTFT at the
            # x-axis context size. Keep them in the table with suffix counts,
            # but only plot full-sync or SpecPrefill recompress events.
            is_full_sync = (
                m.suffix_tokens is not None and
                m.canonical_tokens is not None and
                m.suffix_tokens == m.canonical_tokens
            )
            is_recompress = m.compressed_tokens is not None
            if not (is_full_sync or is_recompress):
                return None
        if attr == "effective_prompt_tps":
            is_full_sync = (
                m.suffix_tokens is not None and
                m.canonical_tokens is not None and
                m.suffix_tokens == m.canonical_tokens
            )
            is_recompress = m.compressed_tokens is not None
            if not (is_full_sync or is_recompress):
                return None
        return getattr(m, attr)

    png_name = "chat_baseline_vs_specprefill.png"
    png_path = out_dir / png_name
    plotted = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), dpi=140, sharex=True)
        specs = [
            ("gen_tps", "Decode tok/s"),
            ("effective_prompt_tps", "Effective prompt tok/s (TTFT throughput)"),
            ("wall_s", "Full turn wall time s"),
        ]
        for ax, (attr, ylabel) in zip(axes, specs):
            for mode in modes:
                xs, ys = [], []
                for turn in turns:
                    m = by_mode_turn.get((mode, turn))
                    y = chart_value(m, attr) if m else None
                    if y is not None:
                        xs.append(x_context(m))
                        ys.append(y)
                if ys:
                    ax.plot(xs, ys, marker="o", label=mode)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Total logical context tokens after turn")
        axes[0].set_title(title)
        axes[0].legend(loc="best")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        plotted = True
        print(f"wrote {png_path}", file=sys.stderr)
    except ImportError:
        pass

    md_path = out_dir / "chat_baseline_vs_specprefill.md"
    lines = [
        f"# {title}",
        "",
        "Generated by `speed-bench/specprefill_chat_loop.py`.",
        "",
        "Real chat-loop test: both modes ran the same prompt sequence while the",
        "logical conversation context grew. This is an actual-usage comparison,",
        "not a cold-prefill sweep at each context length.",
        "",
        "## Measurement Definitions",
        "",
        "- `ctx` = total logical chat context tokens after the turn.",
        "- `target ctx` = DS4 target KV/session tokens after the turn; for SpecPrefill this is compressed.",
        "- `suffix` = tokens actually prefetched/synced for the turn after KV common-prefix reuse.",
        "- `generated` = tokens emitted into the live DS4 session during the turn.",
        "- `TTFT` = chat turn start to first emitted token; DS4 breakdown includes drafter, compression, target prefill, and first decode.",
        "- Default benchmark semantics match mini-01: the model process stays loaded, but `/ctx` resets live KV before each follow-up so each turn prefills the full accumulated transcript.",
        "- Warm follow-up `TTFT` values, when explicitly requested with `--chat-cache warm-repl`, are suffix-prefill timings; TTFT charts omit them unless the turn did a full sync or SpecPrefill recompress.",
        "- `wall time` = full scripted turn time, including the complete generated response.",
        "- `decode tok/s` = emitted tokens / decode elapsed after prefill.",
        "- Chart x-axis is total logical context tokens, matching the mini-01 Qwen revalidation style.",
        "- `--chat-cache warm-repl` keeps live KV across turns and is not the mini-01 comparison.",
        "",
        "## Per-Turn Measurements",
        "",
    ]
    base_label = modes[0] if modes else "baseline"
    sp_label = modes[1] if len(modes) > 1 else (modes[0] if modes else "specprefill")
    header = [
        "turn",
        f"{base_label} ctx",
        f"{base_label} target ctx",
        f"{base_label} suffix",
        f"{base_label} generated",
        f"{base_label} TTFT ms",
        f"{base_label} wall s",
        f"{base_label} decode tok/s",
        f"{sp_label} ctx",
        f"{sp_label} target ctx",
        f"{sp_label} compressed",
        f"{sp_label} suffix",
        f"{sp_label} generated",
        f"{sp_label} TTFT ms",
        f"{sp_label} wall s",
        f"{sp_label} decode tok/s",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")

    def fmt(value, digits=1):
        if value is None:
            return "-"
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.{digits}f}"

    for turn in turns:
        base = by_mode_turn.get((base_label, turn))
        sp = by_mode_turn.get((sp_label, turn))
        row = [
            str(turn + 1),
            fmt(base.transcript_tokens if base else None, 0),
            fmt(base.ctx_session_tokens if base else None, 0),
            fmt(base.suffix_tokens if base else None, 0),
            fmt(base.generated_tokens if base else None, 0),
            fmt(base.ttft_ms if base else None),
            fmt(base.wall_s if base else None),
            fmt(base.gen_tps if base else None),
            fmt(sp.transcript_tokens if sp else None, 0),
            fmt(sp.ctx_session_tokens if sp else None, 0),
            fmt(sp.compressed_tokens if sp else None, 0),
            fmt(sp.suffix_tokens if sp else None, 0),
            fmt(sp.generated_tokens if sp else None, 0),
            fmt(sp.ttft_ms if sp else None),
            fmt(sp.wall_s if sp else None),
            fmt(sp.gen_tps if sp else None),
        ]
        lines.append("| " + " | ".join(row) + " |")

    if plotted:
        lines += ["", "## Chart", "", f"![chart](./{png_name})"]
    lines += [
        "",
        "## Notes",
        "",
        "- Default `--chat-cache cold-full-prompt` matches the mini-01 Qwen harness: reset KV between turns while keeping the model process loaded.",
        "- SpecPrefill cache behavior is controlled by `--spec-prefill-cache`: default `fresh` recompresses the full canonical transcript every turn; `reuse` warm-extends target KV between recompresses.",
        "- Use `metrics.csv` for the TTFT breakdown columns (`drafter_ms`, `target_prefill_ms`, `first_decode_ms`) and suffix/sync counts.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--model", default="./ds4flash.gguf")
    ap.add_argument("--backend",
                    help="Override ds4's default backend (metal/cuda/cpu).  Leave unset to use the default.")
    ap.add_argument("--warm-weights", action="store_true",
                    help="Pass --warm-weights to each ds4 child process.")
    ap.add_argument("--dist-coordinator", action="store_true",
                    help="Run ds4 as a distributed coordinator. Start workers separately first.")
    ap.add_argument("--dist-layers", default="0:19",
                    help="Coordinator layer slice for --dist-coordinator. Default: 0:19.")
    ap.add_argument("--dist-listen-host", default="169.254.149.0",
                    help="Coordinator listen host for --dist-coordinator.")
    ap.add_argument("--dist-listen-port", type=int, default=1234,
                    help="Coordinator listen port for --dist-coordinator.")
    ap.add_argument("--dist-prefill-chunk", type=int, default=0,
                    help="Optional --dist-prefill-chunk value.")
    ap.add_argument("--dist-prefill-window", type=int, default=0,
                    help="Optional --dist-prefill-window value.")
    ap.add_argument("--dist-activation-bits", type=int, default=0,
                    help="Optional --dist-activation-bits value.")
    ap.add_argument("--dist-socket-timeout-sec", type=int, default=0,
                    help="Set DS4_DIST_SOCKET_TIMEOUT_SEC for distributed ds4 child processes.")
    ap.add_argument("--dist-debug", action="store_true",
                    help="Pass --debug to distributed ds4 coordinator.")
    ap.add_argument("--prompt-file",
                    default="tests/long_context_story_prompt.txt",
                    help="First-turn user message (long).  The fixture starts "
                         "with the DSV4 BOS marker so ds4 treats it as a "
                         "pre-rendered chat prompt.")
    ap.add_argument("--turns-file",
                    help="Alternative: one user turn per line.  Overrides "
                         "--prompt-file and --follow-ups.")
    ap.add_argument("--needle-depths",
                    help="Comma-separated needle depths to run as two-turn chat loops, e.g. 0,25,50,75,100.")
    ap.add_argument("--needle-target-chars", type=int, default=64000,
                    help="Approximate long-context needle prompt chars. Default: 64000.")
    ap.add_argument("--needle-seed", type=int, default=20260529,
                    help="Deterministic seed for generated needle filler.")
    ap.add_argument("--follow-ups", nargs="*", default=[
        "Tell me the three most important points so far.",
        "What was the very first fact you saw?",
        "Summarise the conversation in two sentences.",
    ])
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--n-predict", type=int, default=80)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--keep-pct", type=float, default=0.2)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--drafter-score-lookahead", type=int, default=4)
    ap.add_argument("--drafter-score-pool-kernel", type=int, default=13)
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "drafter"],
                    choices=["baseline", "drafter"])
    ap.add_argument("--drafter-model",
                    default=os.environ.get("DRAFTER_MODEL", "./gguf/qwen3.5-0.8b-mlx-4bit"),
                    help="Resident live drafter model for --modes drafter.")
    ap.add_argument("--drafter-backend",
                    default=os.environ.get("DRAFTER_BACKEND", "native"),
                    choices=["native", "python"],
                    help="Drafter backend for SpecPrefill mode. Default native "
                         "keeps scoring in-process; python is the legacy MLX helper.")
    ap.add_argument("--drafter-python",
                    default=os.environ.get("DRAFTER_PYTHON", "python3"))
    ap.add_argument("--drafter-script", default="speed-bench/ds4_live_drafter.py")
    ap.add_argument("--drafter-tokenizer",
                    default=os.environ.get("DRAFTER_TOKENIZER", "./gguf/dsv4-tokenizer"))
    ap.add_argument("--spec-prefill-cache", choices=["fresh", "reuse"], default="fresh",
                    help="Cache mode for --modes drafter. Default fresh matches "
                         "mini-01: score/compress the full canonical transcript "
                         "every turn.")
    ap.add_argument("--chat-cache", choices=["cold-full-prompt", "warm-repl"],
                    default="cold-full-prompt",
                    help="Chat-loop cache semantics. cold-full-prompt resets DS4 "
                         "KV with /ctx before each follow-up while preserving the "
                         "transcript, matching mini-01 --mode cold --chat-loop. "
                         "warm-repl keeps live KV across turns.")
    ap.add_argument("--allow-fresh-drafter-comparison", action="store_true",
                    help="Deprecated no-op retained for old scripts.")
    ap.add_argument("--validate", action="store_true",
                    help="Set DS4_SCORE_VALIDATE=1 so each scoring turn also "
                         "runs the CPU scorer and emits parity diff.")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Per-mode subprocess timeout in seconds (default 30 min)")
    ap.add_argument("--out-dir", default="/tmp/ds4_chatloop")
    ap.add_argument("--report-only-metrics",
                    help="Read an existing metrics.csv and regenerate plots/report without running ds4.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only_metrics:
        all_metrics = read_metrics_csv(Path(args.report_only_metrics))
        if not args.no_plot:
            maybe_plot(all_metrics, out_dir)
            write_comparison_report(
                all_metrics,
                out_dir,
                "DS4 distributed chat-loop comparison — baseline vs SpecPrefill drafter",
            )
        return 0

    if not Path(args.ds4).exists():
        print(f"error: ds4 binary not found at {args.ds4}; run `make` first", file=sys.stderr)
        return 1
    if not Path(args.model).exists():
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return 1

    all_metrics: list[TurnMetrics] = []
    needle_rows: list[dict[str, object]] = []
    mode_runs = build_modes(args)
    if args.needle_depths:
        depths = [int(x.strip()) for x in args.needle_depths.split(",") if x.strip()]
        for depth in depths:
            turns = build_needle_turns(depth, args.needle_target_chars, args.needle_seed)
            print(
                f"needle depth={depth}; turns={len(turns)}; first-turn chars={len(turns[0])}",
                file=sys.stderr,
            )
            for mode in mode_runs:
                stem = f"{mode.label}_d{depth:03d}"
                stderr_path = out_dir / f"{stem}.stderr"
                stdout_path = out_dir / f"{stem}.stdout"
                ms = feed_repl(args, mode, turns, stderr_path, stdout_path, depth=depth)
                all_metrics.extend(ms)
                grade = grade_needle_output(stdout_path.read_text(errors="replace"))
                needle_rows.append({
                    "mode": mode.label,
                    "depth": depth,
                    "stdout_path": str(stdout_path),
                    **grade,
                })
                print(
                    f"  needle {mode.label} depth={depth}: recall={grade['recall_exact']} "
                    f"coherence={grade['coherence']} pass={grade['pass']} stdout={stdout_path}",
                    file=sys.stderr,
                )
                for m in ms:
                    print(
                        f"  turn={m.turn:>2} prompt={m.prompt_tokens} -> "
                        f"compressed={m.compressed_tokens} src={m.scores_source} "
                        f"prefill={m.prefill_tps} t/s gen={m.gen_tps} t/s "
                        f"wall={m.wall_s:.2f}s"
                        + (f" max|m-c|={m.validate_max_abs}" if m.validate_max_abs is not None else ""),
                        file=sys.stderr,
                    )
    else:
        turns = load_turns(args)
        print(f"turns: {len(turns)}; first-turn chars={len(turns[0])}", file=sys.stderr)
        for mode in mode_runs:
            stderr_path = out_dir / f"{mode.label}.stderr"
            stdout_path = out_dir / f"{mode.label}.stdout"
            ms = feed_repl(args, mode, turns, stderr_path, stdout_path)
            all_metrics.extend(ms)
            for m in ms:
                print(
                    f"  turn={m.turn:>2} prompt={m.prompt_tokens} -> "
                    f"compressed={m.compressed_tokens} src={m.scores_source} "
                    f"prefill={m.prefill_tps} t/s gen={m.gen_tps} t/s "
                    f"wall={m.wall_s:.2f}s"
                    + (f" max|m-c|={m.validate_max_abs}" if m.validate_max_abs is not None else ""),
                    file=sys.stderr,
                )

    csv_path = out_dir / "metrics.csv"
    write_csv(all_metrics, csv_path)
    print(f"\nmetrics CSV: {csv_path}", file=sys.stderr)
    if needle_rows:
        needle_csv_path = out_dir / "needle_grades.csv"
        write_needle_csv(needle_rows, needle_csv_path)
        print(f"needle grades CSV: {needle_csv_path}", file=sys.stderr)

    if not args.no_plot:
        maybe_plot(all_metrics, out_dir)
        write_comparison_report(
            all_metrics,
            out_dir,
            "DS4 distributed chat-loop comparison — baseline vs SpecPrefill drafter",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
