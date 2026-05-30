# NanoChat Medical - 3-Stage Training

This project trains one shared-tokenizer medical chatbot in three sequential stages.

## Pipeline

1. **Stage 1 - General text pretraining**
   - Data: climbmix raw text
   - Artifact: `data/pretrain_splits/climbmix_docs.txt`
   - Checkpoint: `data/checkpoints/pre_trained/best_pretrained.pth`

2. **Stage 2 - Medical text continued pretraining**
   - Data: PubMed abstracts plus a small climbmix mix-in
   - Artifact: `data/stage2_splits/stage2_docs.txt`
   - Checkpoint: `data/checkpoints/stage2_checkpoint/best_stage2.pth`

3. **Stage 3 - Chat fine-tuning**
   - Data: oasst2, MedDialog, MedQuAD, emergency cases, and safety examples
   - Artifacts: `data/stage3_splits/train.jsonl`, `val.jsonl`, `test.jsonl`
   - Final checkpoint: `data/checkpoints/best_model.pth`

## Tokenizer

The tokenizer is trained once and shared by all stages.

```bash
python src/chat_model/tokenizing/prepare_data.py
python src/chat_model/tokenizing/train_tokenizer.py
```

Current tokenizer artifact:

```text
data/processed/tokenized/tokenizer.pkl
```

Do not change `VOCAB_SIZE` or retrain the tokenizer after Stage 1 starts unless restarting all training.

## Training

Use the non-interactive training script for Bianca/NAISS SUPR:

```bash
python scripts/train_3stage.py --stage 1
python scripts/train_3stage.py --stage 2
python scripts/train_3stage.py --stage 3
```

Or run all stages sequentially:

```bash
python scripts/train_3stage.py --stage all
```

## Important Config

Check `src/chat_model/config.py` before training:

```python
VOCAB_SIZE = 32_768
N_EMB = 1024
N_LAYER = 8
N_HEAD = 8
BLOCK_SIZE = 1024
```

`N_EMB` must be divisible by `N_HEAD`. Changing model dimensions after training begins makes old checkpoints incompatible.

## Optional Parquet Export

`convert_to_parquet.py` is optional. The current training script reads text and JSONL artifacts directly, so parquet conversion is not required for training.
