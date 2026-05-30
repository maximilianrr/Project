#!/bin/bash
#SBATCH --account=sens2025696
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saugat@kth.se
#SBATCH --job-name=nanochat_train
#SBATCH --output=nanochat_%j.out
#SBATCH --error=nanochat_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -C gpu
#SBATCH --gres=gpu:1

# NanoChat 3-stage training pipeline on Bianca

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/cygnus/proj_nobackup/saugat/health_nanochat/Project_4}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$PROJECT_DIR/../nanochat}"
SIF="${SIF:-/cygnus/proj_nobackup/saugat/health_nanochat/nanochat_train_2.sif}"
LOAD_DATA="${LOAD_DATA:-0}"

DATA_DIR="$PROJECT_DIR/data"
TOKENIZER_PKL="$DATA_DIR/processed/tokenized/tokenizer.pkl"
PRETRAIN_DOCS="$DATA_DIR/pretrain_splits/climbmix_docs.txt"
STAGE2_DOCS="$DATA_DIR/stage2_splits/stage2_docs.txt"
STAGE3_TRAIN="$DATA_DIR/stage3_splits/train.jsonl"
STAGE3_VAL="$DATA_DIR/stage3_splits/val.jsonl"
CHECKPOINTS_DIR="$DATA_DIR/checkpoints"
PRETRAIN_CKPT_DIR="$CHECKPOINTS_DIR/pre_trained"
STAGE2_CKPT_DIR="$CHECKPOINTS_DIR/stage2_checkpoint"
STAGE3_CKPT_DIR="$CHECKPOINTS_DIR/stage3_checkpoint"
LOGS_DIR="$PROJECT_DIR/logs"

echo "============================================================"
echo "  NanoChat 3-stage training - Bianca"
echo "  Job ID      : ${SLURM_JOB_ID:-local}"
echo "  Node        : ${SLURMD_NODENAME:-local}"
echo "  Project dir : $PROJECT_DIR"
echo "  Nanochat dir: $NANOCHAT_DIR"
echo "  Image       : $SIF"
echo "  Load data   : $LOAD_DATA"
echo "  Started     : $(date)"
echo "============================================================"

fail() {
    echo "ERROR: $1"
    exit 1
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1"
        echo "  $2"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing directory: $1"
        echo "  $2"
        exit 1
    fi
}

case "$LOAD_DATA" in
    0|1) ;;
    *) fail "LOAD_DATA must be 0 or 1." ;;
esac

check_dir "$PROJECT_DIR" "Upload/unzip the project as Project_4."
check_dir "$PROJECT_DIR/src/chat_model" "Project_4 must contain src/chat_model."
check_dir "$NANOCHAT_DIR" "Upload/unzip nanochat as a sibling of Project_4."
check_file "$NANOCHAT_DIR/nanochat/tokenizer.py" "The tokenizer pickle needs nanochat.tokenizer.RustBPETokenizer."
check_file "$SIF" "Upload the Singularity image or override SIF=/path/to/image.sif."
check_file "$TOKENIZER_PKL" "Upload tokenizer.pkl to data/processed/tokenized/."

CONTAINER_CMD="${CONTAINER_CMD:-}"
if [ -z "$CONTAINER_CMD" ]; then
    if command -v singularity >/dev/null 2>&1; then
        CONTAINER_CMD="singularity"
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_CMD="apptainer"
    else
        fail "Neither singularity nor apptainer is available on PATH."
    fi
fi

if [ "$LOAD_DATA" = "0" ]; then
    check_file "$PRETRAIN_DOCS" "Upload data/pretrain_splits/climbmix_docs.txt."
    check_file "$STAGE2_DOCS" "Upload data/stage2_splits/stage2_docs.txt."
    check_file "$STAGE3_TRAIN" "Upload data/stage3_splits/train.jsonl."
    check_file "$STAGE3_VAL" "Upload data/stage3_splits/val.jsonl."
else
    check_dir "$DATA_DIR/climbmix_parquet" "Needed to regenerate Stage 1 and Stage 2 splits."
    check_dir "$DATA_DIR/pubmed_parquet" "Needed to regenerate Stage 2 splits."
    check_dir "$DATA_DIR/pretrain_parquet" "Needed to regenerate Stage 3 oasst2 splits."
    check_dir "$DATA_DIR/raw" "Needed to regenerate Stage 3 medical/chat data."
fi

mkdir -p "$LOGS_DIR" "$CHECKPOINTS_DIR" "$PRETRAIN_CKPT_DIR" "$STAGE2_CKPT_DIR" "$STAGE3_CKPT_DIR"

echo "Pre-flight checks passed."
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

TRAIN_ARGS=()
if [ "$LOAD_DATA" = "1" ]; then
    TRAIN_ARGS+=(--load_data)
fi

echo "Launching full 3-stage training..."

set +e
"$CONTAINER_CMD" exec \
    --nv \
    --bind "$PROJECT_DIR:/workspace/Project_4" \
    --bind "$NANOCHAT_DIR:/workspace/nanochat" \
    --bind "${TMPDIR:-/tmp}:/tmp" \
    --env "PYTHONPATH=/workspace/Project_4/src:/workspace/nanochat" \
    --env "HF_DATASETS_OFFLINE=1" \
    --env "TRANSFORMERS_OFFLINE=1" \
    --env "HF_HUB_OFFLINE=1" \
    --env "HF_HOME=/tmp/hf_cache" \
    --env "TOKENIZERS_PARALLELISM=false" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" \
    python -m chat_model.training.train "${TRAIN_ARGS[@]}"

EXIT_CODE=$?
set -e

echo ""
echo "============================================================"
echo "  Finished             : $(date)"
echo "  Exit code            : $EXIT_CODE"
echo "  Stage 1 checkpoints  : $PRETRAIN_CKPT_DIR"
echo "  Stage 2 checkpoints  : $STAGE2_CKPT_DIR"
echo "  Stage 3 checkpoints  : $STAGE3_CKPT_DIR"
echo "  Final best checkpoint: $CHECKPOINTS_DIR/best_model.pth"
echo "============================================================"

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "TRAINING FAILED - check nanochat_${SLURM_JOB_ID:-run}.err"
    exit "$EXIT_CODE"
fi

echo "Done."
