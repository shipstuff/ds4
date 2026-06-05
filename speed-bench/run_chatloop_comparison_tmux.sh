#!/usr/bin/env bash
set -euo pipefail

# Start the baseline-vs-SpecPrefill drafter chat-loop comparison in tmux.
# This is the portable entry point for reproducing the TTFT / effective prompt
# throughput / decode charts on another machine.
#
# Local single-node default:
#   speed-bench/run_chatloop_comparison_tmux.sh
#
# Distributed coordinator mode:
#   DIST=1 COORD_LAYERS=0:19 COORD_LISTEN_HOST=0.0.0.0 COORD_LISTEN_PORT=1234 \
#     speed-bench/run_chatloop_comparison_tmux.sh
#
# Start any distributed workers separately with ./ds4 --role worker.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS4_DIR="${DS4_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

SESSION="${SESSION:-ds4-chatloop-compare}"
LOG="${LOG:-/tmp/ds4-chatloop-compare.log}"
OUT_DIR="${OUT_DIR:-/tmp/ds4_chatloop_compare}"
TMUX_BIN="${TMUX_BIN:-tmux}"

MODEL="${MODEL:-$DS4_DIR/ds4flash.gguf}"
DRAFTER_MODEL="${DRAFTER_MODEL:-$DS4_DIR/gguf/qwen3.5-0.8b-mlx-4bit}"
DRAFTER_TOKENIZER="${DRAFTER_TOKENIZER:-$DS4_DIR/gguf/dsv4-tokenizer}"
DRAFTER_BACKEND="${DRAFTER_BACKEND:-native}"
DRAFTER_PYTHON="${DRAFTER_PYTHON:-python3}"
DRAFTER_SCRIPT="${DRAFTER_SCRIPT:-speed-bench/ds4_live_drafter.py}"

CTX="${CTX:-32768}"
N_PREDICT="${N_PREDICT:-1024}"
TEMP="${TEMP:-0}"
KEEP_PCT="${KEEP_PCT:-0.3}"
SINK="${SINK:-16}"
TAIL="${TAIL:-256}"
CHUNK="${CHUNK:-32}"
TIMEOUT="${TIMEOUT:-3600}"
NEEDLE_DEPTHS="${NEEDLE_DEPTHS:-}"
NEEDLE_TARGET_CHARS="${NEEDLE_TARGET_CHARS:-64000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CHECK_ONLY="${CHECK_ONLY:-0}"

DIST="${DIST:-0}"
COORD_LAYERS="${COORD_LAYERS:-0:19}"
COORD_LISTEN_HOST="${COORD_LISTEN_HOST:-0.0.0.0}"
COORD_LISTEN_PORT="${COORD_LISTEN_PORT:-1234}"
DIST_PREFILL_CHUNK="${DIST_PREFILL_CHUNK:-0}"
DIST_SOCKET_TIMEOUT_SEC="${DIST_SOCKET_TIMEOUT_SEC:-1800}"
DIST_DEBUG="${DIST_DEBUG:-0}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 1
  fi
}

base_cmd=(
  python3 speed-bench/specprefill_chat_loop.py
  --ds4 ./ds4
  --model "$MODEL"
  --modes baseline drafter
  --drafter-model "$DRAFTER_MODEL"
  --drafter-backend "$DRAFTER_BACKEND"
  --ctx "$CTX"
  --n-predict "$N_PREDICT"
  --temp "$TEMP"
  --keep-pct "$KEEP_PCT"
  --sink "$SINK"
  --tail "$TAIL"
  --chunk "$CHUNK"
  --spec-prefill-cache fresh
  --chat-cache cold-full-prompt
  --timeout "$TIMEOUT"
  --out-dir "$OUT_DIR"
)

if [[ "$DRAFTER_BACKEND" == "python" ]]; then
  base_cmd+=(
    --drafter-python "$DRAFTER_PYTHON"
    --drafter-script "$DRAFTER_SCRIPT"
    --drafter-tokenizer "$DRAFTER_TOKENIZER"
  )
elif [[ "$DRAFTER_BACKEND" != "native" ]]; then
  echo "DRAFTER_BACKEND must be native or python, got: $DRAFTER_BACKEND" >&2
  exit 1
fi

if [[ -n "$NEEDLE_DEPTHS" ]]; then
  base_cmd+=(--needle-depths "$NEEDLE_DEPTHS" --needle-target-chars "$NEEDLE_TARGET_CHARS")
fi

if [[ "$DIST" == "1" ]]; then
  base_cmd+=(
    --dist-coordinator
    --dist-layers "$COORD_LAYERS"
    --dist-listen-host "$COORD_LISTEN_HOST"
    --dist-listen-port "$COORD_LISTEN_PORT"
    --dist-socket-timeout-sec "$DIST_SOCKET_TIMEOUT_SEC"
  )
  if [[ "$DIST_PREFILL_CHUNK" != "0" ]]; then
    base_cmd+=(--dist-prefill-chunk "$DIST_PREFILL_CHUNK")
  fi
  if [[ "$DIST_DEBUG" == "1" ]]; then
    base_cmd+=(--dist-debug)
  fi
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_split=($EXTRA_ARGS)
  base_cmd+=("${extra_split[@]}")
fi

printf -v command_str "%q " "${base_cmd[@]}"
run_cmd="cd \"$DS4_DIR\"; $command_str 2>&1 | tee \"$LOG\""

cat <<EOF
chat-loop comparison
  tmux session: $SESSION
  log:          $LOG
  out dir:      $OUT_DIR
  model:        $MODEL
  drafter:      $DRAFTER_MODEL
  backend:      $DRAFTER_BACKEND
  ctx:          $CTX
  n_predict:    $N_PREDICT
  distributed:  $DIST
EOF

if [[ "$CHECK_ONLY" == "1" ]]; then
  printf "\ncommand:\n%s\n" "$run_cmd"
  exit 0
fi

require_path "$DS4_DIR/ds4" "ds4 binary"
require_path "$MODEL" "target model"
require_path "$DRAFTER_MODEL" "drafter model"
if [[ "$DRAFTER_BACKEND" == "python" ]]; then
  require_path "$DRAFTER_TOKENIZER" "DSV4 tokenizer"
  require_path "$DS4_DIR/$DRAFTER_SCRIPT" "drafter helper"
fi

"$TMUX_BIN" kill-session -t "$SESSION" 2>/dev/null || true
"$TMUX_BIN" new-session -d -s "$SESSION" "$run_cmd"

cat <<EOF

started
  attach: $TMUX_BIN attach -t $SESSION
  tail:   tail -f $LOG
EOF
