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

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/cygnus/proj_nobackup/saugat/health_nanochat/Project_4}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$PROJECT_DIR/../nanochat}"
SIF="${SIF:-/cygnus/proj_nobackup/saugat/health_nanochat/nanochat_train_2.sif}"

DATA_DIR="$PROJECT_DIR/data"

check_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "Missing directory: $1"
        exit 1
    fi
}

check_dir "$PROJECT_DIR"
check_dir "$PROJECT_DIR/src/chat_model"
check_dir "$NANOCHAT_DIR"
check_file "$NANOCHAT_DIR/nanochat/tokenizer.py"
check_file "$SIF"
check_file "$DATA_DIR/processed/tokenized/tokenizer.pkl"
check_file "$DATA_DIR/pretrain_splits/climbmix_docs.txt"
check_file "$DATA_DIR/stage2_splits/stage2_docs.txt"
check_file "$DATA_DIR/stage3_splits/train.jsonl"
check_file "$DATA_DIR/stage3_splits/val.jsonl"

mkdir -p "$DATA_DIR/checkpoints/pre_trained"
mkdir -p "$DATA_DIR/checkpoints/stage2_checkpoint"
mkdir -p "$DATA_DIR/checkpoints/stage3_checkpoint"

echo "Starting NanoChat training"
echo "Project: $PROJECT_DIR"
echo "Nanochat: $NANOCHAT_DIR"
echo "Started: $(date)"

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:/workspace/Project_4" \
    --bind "$NANOCHAT_DIR:/workspace/nanochat" \
    --bind "${TMPDIR:-/tmp}:/tmp" \
    --env "PYTHONPATH=/workspace/Project_4/src:/workspace/nanochat" \
    --env "TOKENIZERS_PARALLELISM=false" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" \
    python -m chat_model.training.train

echo "Finished: $(date)"
echo "Final model should be at: $DATA_DIR/checkpoints/best_model.pth"
