#!/bin/bash
#SBATCH --account=sens2025696
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=saugat@kth.se
#SBATCH --job-name=nanochat_train
#SBATCH --output=logs/nanochat_%j.out
#SBATCH --error=logs/nanochat_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -C gpu
#SBATCH --gres=gpu:1

# NanoChat — Full training pipeline on Bianca (NAISS SUPR)
# Account : sensxxxxxxx  |  User: USER
#
# Stages:
#   Stage 1 — oasst2 general conversational pretraining
#   Stage 2 — medical fine-tuning
#
# Control which stages run via TRAIN_MODE (default: both):
#   TRAIN_MODE=pretrain  sbatch run_training.sh   # Stage 1 only
#   TRAIN_MODE=finetune  sbatch run_training.sh   # Stage 2 only
#   TRAIN_MODE=both      sbatch run_training.sh   # Both (default)
#
# Submit:   sbatch run_training.sh
# Monitor:  tail -f logs/nanochat_<JOBID>.out

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/cygnus/proj_nobackup/saugat/health_nanochat/Project_3}"
NANOCHAT_DIR="$(realpath "$PROJECT_DIR/../nanochat")"
SIF="${SIF:-/cygnus/proj_nobackup/saugat/health_nanochat/nanochat_train_2.sif}"
TRAIN_MODE="${TRAIN_MODE:-both}"       # pretrain | finetune | both

# Derived paths — must match config.py exactly
DATA_DIR="$PROJECT_DIR/data"
SPLITS_DIR="$DATA_DIR/splits"                         # finetune splits (JSONL)
PRETRAIN_SPLITS_DIR="$DATA_DIR/pretrain_splits"       # pretrain splits (JSONL)
TOKENIZER_PKL="$DATA_DIR/processed/tokenized/tokenizer.pkl"
PRE_TRAINED_DIR="$DATA_DIR/checkpoints/pre_trained"  # Stage 1 checkpoint dir
CHECKPOINTS_DIR="$DATA_DIR/checkpoints"              # Stage 2 checkpoint dir
LOGS_DIR="$PROJECT_DIR/logs"

# Pre-flight checks
echo "============================================================"
echo "  NanoChat Training — Bianca"
echo "  Job ID   : ${SLURM_JOB_ID:-local}"
echo "  Node     : ${SLURMD_NODENAME:-local}"
echo "  Mode     : $TRAIN_MODE"
echo "  Started  : $(date)"
echo "============================================================"

check() {
    if [ ! -e "$1" ]; then
        echo "ERROR: Not found — $1"
        echo "  $2"
        exit 1
    fi
}

check "$SIF"            "Build nanochat_train_2.sif and transfer via wharf."
check "$PROJECT_DIR"    "Unzip Project-2.zip, or set PROJECT_DIR env var."
check "$NANOCHAT_DIR"   "Unzip nanochat.zip as a sibling of Project-2/."
check "$TOKENIZER_PKL"  "Transfer tokenizer.pkl into data/processed/tokenized/ via wharf."

# Check splits depending on mode
if [[ "$TRAIN_MODE" == "pretrain" || "$TRAIN_MODE" == "both" ]]; then
    check "$PRETRAIN_SPLITS_DIR/train.jsonl" "Transfer pretrain_splits/ into $DATA_DIR/ via wharf."
    check "$PRETRAIN_SPLITS_DIR/val.jsonl"   "Transfer pretrain_splits/ into $DATA_DIR/ via wharf."
fi

if [[ "$TRAIN_MODE" == "finetune" || "$TRAIN_MODE" == "both" ]]; then
    check "$SPLITS_DIR/train.jsonl" "Transfer splits/ (finetune) into $DATA_DIR/ via wharf."
    check "$SPLITS_DIR/val.jsonl"   "Transfer splits/ (finetune) into $DATA_DIR/ via wharf."
fi

if [[ "$TRAIN_MODE" == "finetune" ]]; then
    check "$PRE_TRAINED_DIR/best_pretrained.pth" \
        "Run Stage 1 first (--mode pretrain) to generate pretrained weights."
fi

echo "All pre-flight checks passed."

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "$PRE_TRAINED_DIR" "$CHECKPOINTS_DIR" "$LOGS_DIR"

# ── Load Singularity module 
# module load singularity

# GPU info
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Launch training
echo "Launching training (mode: $TRAIN_MODE)"

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:/workspace/Project_3" \
    --bind "$NANOCHAT_DIR:/workspace/nanochat" \
    --bind "${TMPDIR:-/tmp}:/tmp" \
    --env "PYTHONPATH=/workspace/Project_3/src:/workspace/nanochat" \
    --env "HF_DATASETS_OFFLINE=1" \
    --env "TRANSFORMERS_OFFLINE=1" \
    --env "HF_HUB_OFFLINE=1" \
    --env "HF_HOME=/tmp/hf_cache" \
    --env "TOKENIZERS_PARALLELISM=false" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" \
    python /workspace/Project_3/run_bianca.py \
        --mode "$TRAIN_MODE"

EXIT_CODE=$?

# Result
echo ""
echo "============================================================"
echo "  Finished  : $(date)"
echo "  Exit code : $EXIT_CODE"
echo "  Mode      : $TRAIN_MODE"
echo "  Stage 1 checkpoints : $PRE_TRAINED_DIR"
echo "  Stage 2 checkpoints : $CHECKPOINTS_DIR"
echo "============================================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "TRAINING FAILED — check logs/nanochat_${SLURM_JOB_ID:-run}.err"
    exit $EXIT_CODE
fi

echo "Done."
