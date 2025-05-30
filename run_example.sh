# Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -c <ckpt_dir> [-g <num_gpus>]
  -c  Path to your model checkpoint directory
  -g  Number of GPUs to use (defaults to all available GPUs)
  -p  Path to prompt file
  -o  Path to output location
EOF
  exit 1
}

OUTPUT_DIR="samples"
# parse args
CKPT_DIR=""
PROMPT="examples/test.yaml"
NGPUS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--ckpt_dir)
      CKPT_DIR="$2"; shift 2;;
    -g|--gpus)
      NGPUS="$2"; shift 2;;
    -p|--prompt)
      PROMPT="$2"; shift 2;;
    -o|--output)
      OUTPUT_DIR="$2"; shift 2;;
    -*)
      echo "Unknown option: $1" >&2; usage;;
    *)
      break;;
  esac
done


if [[ -z "$CKPT_DIR" ]]; then
  echo "Error: --ckpt_dir is required" >&2
  usage
fi

# detect GPUs if not provided
if [[ -z "$NGPUS" ]]; then
  if command -v python3 &>/dev/null; then
    NGPUS=$(python3 - <<'PYCODE'
import torch
print(torch.cuda.device_count() or 1)
PYCODE
)
  else
    echo "Warning: python3 not found; defaulting to 1 GPU" >&2
    NGPUS=1
  fi
fi

echo ">>> Using checkpoint: $CKPT_DIR"
echo ">>> Generate case: $PROMPT"
echo ">>> Saved to: $OUTPUT_DIR"
echo ">>> Detected $NGPUS GPU(s)"

mkdir -p $OUTPUT_DIR/outputs

if [[ "$NGPUS" -eq 1 ]]; then
  echo ">>> Single‐GPU mode: running generate.py directly"
  python generate.py \
    --ckpt_dir "$CKPT_DIR" \
    --prompt $PROMPT \
    --save_file "$OUTPUT_DIR/outputs/%03d.mp4"
else
  echo ">>> Multi‐GPU mode: launching with torchrun"
  torchrun \
    --nproc_per_node="$NGPUS" \
    --master-port=5645 \
    generate.py \
      --ckpt_dir "$CKPT_DIR" \
      --prompt $PROMPT \
      --save_file "$OUTPUT_DIR/outputs/%03d.mp4" \
      --ulysses_size "$NGPUS" \
      --base_seed 4567 \
      --dit_fsdp \
      --t5_fsdp
fi

cp $PROMPT "$OUTPUT_DIR/" &

# visualize results
python3 ./tools/visualize_trajectory.py --base_dir "$OUTPUT_DIR/"
python3 ./tools/plot_user_inputs.py $PROMPT --save_dir $OUTPUT_DIR/image_with_tracks
