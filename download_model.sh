#!/bin/sh
set -e

REPO="antirez/deepseek-v4-gguf"
TOKENIZER_REPO="deepseek-ai/DeepSeek-V4-Flash"
DRAFTER_REPO="mlx-community/Qwen3.5-0.8B-4bit"
Q2_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf"
Q2_IMATRIX_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q4_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf"
Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
Q2_Q4_IMATRIX_FILE="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
PRO_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct.gguf"
PRO_IMATRIX_FILE="DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf"
MTP_FILE="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
TOKENIZER_FILES="config.json tokenizer.json tokenizer_config.json"
DRAFTER_FILES="config.json model.safetensors model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json chat_template.jinja preprocessor_config.json processor_config.json video_preprocessor_config.json"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${DS4_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
DRAFTER_DIR=${DS4_DRAFTER_DIR:-"$OUT_DIR/qwen3.5-0.8b-mlx-4bit"}
TOKENIZER_DIR=${DS4_TOKENIZER_DIR:-"$OUT_DIR/dsv4-tokenizer"}
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
DeepSeek V4 GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh q4-imatrix [--token TOKEN]
  ./download_model.sh q2 [--token TOKEN]
  ./download_model.sh q4 [--token TOKEN]
  ./download_model.sh pro [--token TOKEN]
  ./download_model.sh pro-imatrix [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]
  ./download_model.sh drafter [--token TOKEN]
  ./download_model.sh specprefill [--token TOKEN]

Targets:
  *** PREFERRED GGUF FILES: USE THE IMATRIX VERSIONS BELOW ***

  q2-imatrix
       2-bit routed experts, about 81 GB on disk.
       Recommended model for 96 and 128 GB RAM machines.

  q2-q4-imatrix
       Mixed Flash quant: mostly q2 routed experts, with the last 6 layers
       using q4 routed experts. About 98 GB on disk.

  q4-imatrix
       4-bit routed experts, about 153 GB on disk.
       Recommended model for machines with 256 GB RAM or more.

  pro-imatrix
       DeepSeek V4 PRO imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines.

  Legacy GGUF files:

  q2   2-bit routed experts, about 81 GB on disk.
       Older non-imatrix model for 96 and 128 GB RAM machines. Prefer
       q2-imatrix unless you specifically need the legacy quant.

  q4   4-bit routed experts, about 153 GB on disk.
       Older non-imatrix model for machines with 256 GB RAM or more. Prefer
       q4-imatrix unless you specifically need the legacy quant.

  pro  DeepSeek V4 PRO non-imatrix quant, as a single GGUF file. About 430 GB
       on disk; intended for 512 GB RAM machines. Prefer pro-imatrix unless you
       specifically need the legacy quant.

  mtp  Optional speculative decoding component, about 3.5 GB on disk.
       It is useful with q2-imatrix, q4-imatrix, q2, and q4, but must be
       enabled explicitly with --mtp when running ds4 or ds4-server.

  drafter
       Download only the DSV4 tokenizer plus the MLX/Qwen drafter used by
       SpecPrefill. Use this when the target GGUF already exists.

  specprefill
       Download q2-imatrix, the DSV4 tokenizer, and the MLX/Qwen drafter.
       This is the single-node SpecPrefill setup path for 96 GB+ machines.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF token
                 cache is used if present.

Environment:
  DS4_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf
  DS4_DRAFTER_DIR
                 Directory used for the MLX/Qwen drafter.
                 Default: ./gguf/qwen3.5-0.8b-mlx-4bit
  DS4_TOKENIZER_DIR
                 Directory used for the DSV4 tokenizer files.
                 Default: ./gguf/dsv4-tokenizer

After main-model downloads the script updates:
  ./ds4flash.gguf -> <download directory>/<selected model>

Then the default commands work:
  ./ds4 -p "Hello"
  ./ds4-server --ctx 100000

After downloading mtp, enable it explicitly, for example:
  ./ds4 --mtp <download directory>/$MTP_FILE --mtp-draft 2
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_IMATRIX_FILE ;;
    q4-imatrix) MODEL_FILE=$Q4_IMATRIX_FILE ;;
    q2) MODEL_FILE=$Q2_FILE ;;
    q4) MODEL_FILE=$Q4_FILE ;;
    pro) MODEL_FILE=$PRO_FILE ;;
    pro-imatrix) MODEL_FILE=$PRO_IMATRIX_FILE ;;
    mtp) MODEL_FILE=$MTP_FILE ;;
    drafter) MODEL_FILE= ;;
    specprefill) MODEL_FILE=$Q2_IMATRIX_FILE ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    repo=$1
    dir=$2
    file=$3
    out="$dir/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$repo/resolve/main/$file"

    mkdir -p "$dir"

    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $file"
    echo "from https://huggingface.co/$repo"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_specprefill_assets() {
    for file in $TOKENIZER_FILES; do
        download_one "$TOKENIZER_REPO" "$TOKENIZER_DIR" "$file"
    done
    for file in $DRAFTER_FILES; do
        download_one "$DRAFTER_REPO" "$DRAFTER_DIR" "$file"
    done
}

if [ -n "$MODEL_FILE" ]; then
    download_one "$REPO" "$OUT_DIR" "$MODEL_FILE"
fi

if [ "$MODEL" = "drafter" ] || [ "$MODEL" = "specprefill" ]; then
    download_specprefill_assets
fi

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix, q4-imatrix, q2, and q4."
    echo "Enable it explicitly, for example:"
    echo "  ./ds4 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
elif [ "$MODEL" != "drafter" ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
    echo "Linked ./ds4flash.gguf -> $OUT_DIR/$MODEL_FILE"
fi

if [ "$MODEL" = "drafter" ] || [ "$MODEL" = "specprefill" ]; then
    echo
    echo "SpecPrefill assets:"
    echo "  tokenizer: $TOKENIZER_DIR"
    echo "  drafter:   $DRAFTER_DIR"
fi

echo
echo "Done."
