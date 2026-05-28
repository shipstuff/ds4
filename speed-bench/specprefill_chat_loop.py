#!/usr/bin/env python3
"""Drive ./ds4 through a scripted multi-turn chat, capture per-turn
SpecPrefill / timing metrics, write a CSV, and plot the comparison.

Designed for the SpecPrefill experimental flag added in
`seslly/specprefill`.  Runs the same conversation under up to three
modes (baseline / heuristic / self-score) and surfaces:

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
      --modes baseline heuristic selfscore \\
      --validate \\
      --out-dir /tmp/ds4_chatloop

Without --validate the script does NOT touch DS4_SCORE_VALIDATE, so the
runs are at full speed and you only get the CSV metrics + plots.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ds4's own log lines.  These come from ds4_cli.c (DS4_LOG_PREFILL,
# DS4_LOG_TIMING) and from ds4.c (DS4_SCORE_VALIDATE path).  Anchored on
# the parts that don't change run-to-run.
RE_SPECPREFILL = re.compile(
    r"spec-prefill:\s*prompt\s*(\d+)\s*->\s*(\d+)\s*tokens\s*"
    r"\(keep=([\d.]+)\s*tail=(\d+)\s*chunk=(\d+)\s*scores=(\S+)\)"
)
RE_TIMING = re.compile(
    r"prefill:\s*([\d.]+)\s*t/s,\s*generation:\s*([\d.]+)\s*t/s"
)
RE_VALIDATE = re.compile(
    r"spec-prefill validate:.*?max\|m-c\|=([\d.]+).*?mean\|m-c\|=([\d.]+)\s*rms=([\d.]+)"
)


@dataclass
class TurnMetrics:
    mode: str
    turn: int
    prompt_tokens: int | None = None
    compressed_tokens: int | None = None
    scores_source: str | None = None
    prefill_tps: float | None = None
    gen_tps: float | None = None
    validate_max_abs: float | None = None
    validate_rms: float | None = None
    wall_s: float | None = None


@dataclass
class ModeRun:
    label: str
    args_extra: list[str] = field(default_factory=list)


def build_modes(args) -> list[ModeRun]:
    modes: list[ModeRun] = []
    if "baseline" in args.modes:
        modes.append(ModeRun("baseline", []))
    if "heuristic" in args.modes:
        modes.append(ModeRun(
            "heuristic",
            [
                f"--spec-prefill={args.keep_pct}",
                "--spec-prefill-tail", str(args.tail),
                "--spec-prefill-chunk", str(args.chunk),
            ],
        ))
    if "selfscore" in args.modes:
        modes.append(ModeRun(
            "selfscore",
            [
                f"--spec-prefill={args.keep_pct}",
                "--spec-prefill-tail", str(args.tail),
                "--spec-prefill-chunk", str(args.chunk),
                "--spec-prefill-self-score",
                "--spec-prefill-score-layers", str(args.score_layers),
                "--spec-prefill-score-lookahead", str(args.score_lookahead),
            ],
        ))
    return modes


def load_turns(args) -> list[str]:
    """Build the multi-turn script.  Turn 1 is the long prompt; subsequent
    turns are short follow-ups that exercise the model's memory of the
    compressed history."""
    if args.turns_file:
        # One turn per non-empty line.
        return [t.rstrip("\n") for t in Path(args.turns_file).read_text().splitlines() if t.strip()]
    first = Path(args.prompt_file).read_text()
    return [first.replace("\n", " ").strip()] + list(args.follow_ups)


def feed_repl(args, mode: ModeRun, turns: list[str], stderr_path: Path) -> list[TurnMetrics]:
    """Run ds4 once with the given mode flags, pipe each turn through stdin,
    parse stderr live to attribute log lines to turn indices."""
    base_args = [
        args.ds4,
        "-m", args.model,
        "--ctx", str(args.ctx),
        "--nothink",
        "-n", str(args.n_predict),
    ]
    if args.backend:
        base_args += ["--backend", args.backend]
    cmd = base_args + mode.args_extra
    env = os.environ.copy()
    if args.validate:
        env["DS4_SCORE_VALIDATE"] = "1"

    stdin_text = "\n".join(turns) + "\n/quit\n"

    print(f"\n=== {mode.label} ===", file=sys.stderr)
    print("  " + " ".join(cmd), file=sys.stderr)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )
    # We need to know per-turn wall time.  Feed each turn individually,
    # measuring between turn-input and the "prefill: X t/s, generation: Y
    # t/s" log line that marks the end of that turn.
    turns_metrics: list[TurnMetrics] = []
    current = TurnMetrics(mode=mode.label, turn=0)
    turn_idx = 0
    stderr_lines: list[str] = []
    t_started: float | None = None

    def flush_current(commit: bool = True) -> None:
        nonlocal current, turn_idx
        if commit and current.prefill_tps is not None:
            turns_metrics.append(current)
        turn_idx += 1
        current = TurnMetrics(mode=mode.label, turn=turn_idx)

    # Send each turn, then drain stderr lines until the timing line for
    # that turn appears or the process exits.
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for i, turn in enumerate(turns):
            t_started = time.perf_counter()
            current = TurnMetrics(mode=mode.label, turn=i)
            proc.stdin.write(turn + "\n")
            proc.stdin.flush()
            # Block-read stderr until we see the timing line for this turn.
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                stderr_lines.append(line)
                m = RE_SPECPREFILL.search(line)
                if m:
                    current.prompt_tokens = int(m.group(1))
                    current.compressed_tokens = int(m.group(2))
                    current.scores_source = m.group(6)
                m = RE_VALIDATE.search(line)
                if m:
                    current.validate_max_abs = float(m.group(1))
                    current.validate_rms = float(m.group(3))
                m = RE_TIMING.search(line)
                if m:
                    current.prefill_tps = float(m.group(1))
                    current.gen_tps = float(m.group(2))
                    current.wall_s = time.perf_counter() - (t_started or 0.0)
                    turns_metrics.append(current)
                    break
        # Send /quit and drain remaining output.
        proc.stdin.write("/quit\n")
        proc.stdin.close()
        for line in proc.stderr.read().splitlines(keepends=True):
            stderr_lines.append(line)
    finally:
        try:
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr_path.write_text("".join(stderr_lines))

    return turns_metrics


def write_csv(metrics: list[TurnMetrics], out_path: Path) -> None:
    fields = [
        "mode", "turn",
        "prompt_tokens", "compressed_tokens", "scores_source",
        "prefill_tps", "gen_tps", "wall_s",
        "validate_max_abs", "validate_rms",
    ]
    with out_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(fields)
        for m in metrics:
            w.writerow([getattr(m, f) for f in fields])


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
    turns = sorted({m.turn for m in metrics})

    def series(mode: str, attr: str) -> list[float]:
        by_turn = {m.turn: getattr(m, attr) for m in metrics if m.mode == mode}
        return [by_turn.get(t) for t in turns]

    plots = [
        ("wall_s",            "Per-turn wall time (s)",       "lower is better"),
        ("prefill_tps",       "Prefill throughput (t/s)",     "higher is better"),
        ("gen_tps",           "Generation throughput (t/s)",  "higher is better"),
        ("compressed_tokens", "Compressed prompt tokens",     "lower = more compression"),
    ]

    for attr, title, hint in plots:
        fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
        for mode in modes:
            ys = series(mode, attr)
            xs = [t for t, y in zip(turns, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if not ys:
                continue
            ax.plot(xs, ys, marker="o", label=mode)
        ax.set_xlabel("Turn index")
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
            ys = series(mode, "validate_max_abs")
            xs = [t for t, y in zip(turns, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if not ys:
                continue
            ax.plot(xs, ys, marker="o", label=mode)
        ax.set_xlabel("Turn index")
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--model", default="./ds4flash.gguf")
    ap.add_argument("--backend",
                    help="Override ds4's default backend (metal/cuda/cpu).  Leave unset to use the default.")
    ap.add_argument("--prompt-file",
                    default="tests/long_context_story_prompt.txt",
                    help="First-turn user message (long).  The fixture starts "
                         "with the DSV4 BOS marker so ds4 treats it as a "
                         "pre-rendered chat prompt.")
    ap.add_argument("--turns-file",
                    help="Alternative: one user turn per line.  Overrides "
                         "--prompt-file and --follow-ups.")
    ap.add_argument("--follow-ups", nargs="*", default=[
        "Tell me the three most important points so far.",
        "What was the very first fact you saw?",
        "Summarise the conversation in two sentences.",
    ])
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--n-predict", type=int, default=80)
    ap.add_argument("--keep-pct", type=float, default=0.3)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--score-layers", type=int, default=2)
    ap.add_argument("--score-lookahead", type=int, default=4)
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "heuristic", "selfscore"],
                    choices=["baseline", "heuristic", "selfscore"])
    ap.add_argument("--validate", action="store_true",
                    help="Set DS4_SCORE_VALIDATE=1 so each scoring turn also "
                         "runs the CPU scorer and emits parity diff.")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="Per-mode subprocess timeout in seconds (default 30 min)")
    ap.add_argument("--out-dir", default="/tmp/ds4_chatloop")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.ds4).exists():
        print(f"error: ds4 binary not found at {args.ds4}; run `make` first", file=sys.stderr)
        return 1
    if not Path(args.model).exists():
        print(f"error: model not found at {args.model}", file=sys.stderr)
        return 1

    turns = load_turns(args)
    print(f"turns: {len(turns)}; first-turn chars={len(turns[0])}", file=sys.stderr)

    all_metrics: list[TurnMetrics] = []
    for mode in build_modes(args):
        stderr_path = out_dir / f"{mode.label}.stderr"
        ms = feed_repl(args, mode, turns, stderr_path)
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

    if not args.no_plot:
        maybe_plot(all_metrics, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
