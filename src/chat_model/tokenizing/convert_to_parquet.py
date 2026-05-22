# src/chat_model/tokenizing/convert_to_parquet.py
"""Convert tokenizer_text.txt and medical JSONL splits to parquet shards.

Produces three output directories:
  data/processed/parquet/          ← tokenizer training shards (from tokenizer_text.txt)
  data/processed/pretrain_parquet/ ← Stage 1 oasst2 + PubMed fine-tune shards
  data/processed/finetune_parquet/ ← Stage 2 medical fine-tuning shards
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chat_model import config

TEXT_SCHEMA = pa.schema([pa.field("text", pa.string())])


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _write_shards(rows: list[str], out_dir: Path, prefix: str) -> tuple[int, int]:
    """Write rows as parquet shards. Returns (total_rows, total_chars) written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards      = [rows[i:i + config.SHARD_SIZE] for i in range(0, len(rows), config.SHARD_SIZE)]
    total_chars = 0
    for i, shard in enumerate(shards):
        out_path    = out_dir / f"{prefix}_{i:05d}.parquet"
        table       = pa.table({"text": shard}, schema=TEXT_SCHEMA)
        pq.write_table(table, out_path, compression="snappy")
        shard_chars  = sum(len(r) for r in shard)
        total_chars += shard_chars
        tag = "  ← validation shard (nanochat convention)" if i == len(shards) - 1 else ""
        print(f"  {out_path.name:<32} {len(shard):>8,} rows  {shard_chars / 1e6:>6.1f} MB{tag}")
    return len(rows), total_chars


def _validate_conversation(raw: str) -> str | None:
    try:
        conv = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(conv, list) or len(conv) < 2:
        return None
    for msg in conv:
        if not isinstance(msg, dict):
            return None
        if msg.get("role") not in {"user", "assistant", "system"}:
            return None
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            return None
    return json.dumps(conv, ensure_ascii=False)


def _convert_tokenizer_shards() -> tuple[int, int]:
    """Convert tokenizer_text.txt → parquet shards for tokenizer training."""
    tok_path = Path(config.TOKENIZER_TEXT)
    if not tok_path.exists():
        raise FileNotFoundError(
            f"{tok_path} not found.\n"
            "Run: prepare_tokenizer_data() first."
        )

    print(f"Reading {tok_path}...")
    with tok_path.open(encoding="utf-8") as f:
        rows = [_normalise(line) for line in f if line.strip()]
    print(f"  Total lines: {len(rows):,}\n")

    print(f"Saving tokenizer shards to {config.PARQUET_DIR}/")
    return _write_shards(rows, Path(config.PARQUET_DIR), "shard")


def _convert_pretrain_splits() -> tuple[int, int]:
    """Convert oasst2 + PubMed JSONL splits → parquet shards for Stage 1 training."""
    pretrain_dir = Path(config.PRETRAIN_SPLITS_DIR)
    out_dir      = Path(config.PRETRAIN_PARQUET_DIR)
    total_rows   = total_chars = 0

    print(f"\nSaving pretrain shards to {out_dir}/")

    for split in ("train", "val", "test"):
        path = pretrain_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"  {split}: missing {path} — skipped (run preprocess_pretrain() first)")
            continue

        valid_rows = []
        with path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                validated = _validate_conversation(raw)
                if validated:
                    valid_rows.append(validated)

        if not valid_rows:
            print(f"  {split}: no valid rows — skipped")
            continue

        if len(valid_rows) <= config.SHARD_SIZE:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path    = out_dir / f"{split}.parquet"
            table       = pa.table({"text": valid_rows}, schema=TEXT_SCHEMA)
            pq.write_table(table, out_path, compression="snappy")
            shard_chars  = sum(len(r) for r in valid_rows)
            total_rows  += len(valid_rows)
            total_chars += shard_chars
            print(f"  {out_path.name:<32} {len(valid_rows):>8,} rows  {shard_chars / 1e6:>6.1f} MB")
        else:
            rows, chars  = _write_shards(valid_rows, out_dir, split)
            total_rows  += rows
            total_chars += chars

    return total_rows, total_chars


def _convert_finetune_splits() -> tuple[int, int]:
    """Convert medical JSONL splits → parquet shards for Stage 2 fine-tuning."""
    finetune_dir = Path(config.FINETUNE_PARQUET_DIR)
    total_rows   = total_chars = 0

    print(f"\nSaving fine-tune shards to {finetune_dir}/")

    for split in ("train", "val", "test"):
        path = Path(config.SPLITS_DIR) / f"{split}.jsonl"
        if not path.exists():
            print(f"  {split}: missing {path} — skipped (run start_preprocess() first)")
            continue

        valid_rows = []
        with path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                validated = _validate_conversation(raw)
                if validated:
                    valid_rows.append(validated)

        if not valid_rows:
            print(f"  {split}: no valid rows — skipped")
            continue

        if len(valid_rows) <= config.SHARD_SIZE:
            finetune_dir.mkdir(parents=True, exist_ok=True)
            out_path    = finetune_dir / f"{split}.parquet"
            table       = pa.table({"text": valid_rows}, schema=TEXT_SCHEMA)
            pq.write_table(table, out_path, compression="snappy")
            shard_chars  = sum(len(r) for r in valid_rows)
            total_rows  += len(valid_rows)
            total_chars += shard_chars
            print(f"  {out_path.name:<32} {len(valid_rows):>8,} rows  {shard_chars / 1e6:>6.1f} MB")
        else:
            rows, chars  = _write_shards(valid_rows, finetune_dir, split)
            total_rows  += rows
            total_chars += chars

    return total_rows, total_chars


def convert_to_parquet() -> None:
    """Convert all pipeline outputs to parquet."""

    print("\nConverting pipeline outputs to parquet")
    print("=" * 48)

    # 1. Tokenizer shards
    tok_rows, tok_chars = _convert_tokenizer_shards()

    # 2. Stage 1 pretrain shards (oasst2 + PubMed JSONL splits)
    pre_rows, pre_chars = _convert_pretrain_splits()

    # 3. Stage 2 fine-tune shards (medical JSONL splits)
    ft_rows, ft_chars = _convert_finetune_splits()

    # Summary
    print("\nSummary")
    print("=" * 48)
    print(f"  Tokenizer shards : {config.PARQUET_DIR}/")
    print(f"    {tok_rows:>10,} rows   {tok_chars / 1e6:>8.1f} MB")
    print(f"  Pretrain shards  : {config.PRETRAIN_PARQUET_DIR}/")
    print(f"    {pre_rows:>10,} rows   {pre_chars / 1e6:>8.1f} MB")
    print(f"  Fine-tune shards : {config.FINETUNE_PARQUET_DIR}/")
    print(f"    {ft_rows:>10,} rows   {ft_chars / 1e6:>8.1f} MB")
    total_chars = tok_chars + pre_chars + ft_chars
    print(f"\n  Total written    : {total_chars / 1e6:.1f} MB")
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert pipeline outputs to parquet shards.")
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer shard conversion (tokenizer_text.txt → parquet).",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip Stage 1 pretrain shard conversion (oasst2 + PubMed JSONL → parquet).",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip Stage 2 fine-tune shard conversion (medical JSONL → parquet).",
    )
    args = parser.parse_args()

    print("\nConverting pipeline outputs to parquet")
    print("=" * 48)

    tok_rows = tok_chars = 0
    pre_rows = pre_chars = 0
    ft_rows  = ft_chars  = 0

    if not args.skip_tokenizer:
        tok_rows, tok_chars = _convert_tokenizer_shards()
    else:
        print("Skipping tokenizer shards.")

    if not args.skip_pretrain:
        pre_rows, pre_chars = _convert_pretrain_splits()
    else:
        print("Skipping pretrain shards.")

    if not args.skip_finetune:
        ft_rows, ft_chars = _convert_finetune_splits()
    else:
        print("Skipping fine-tune shards.")

    print("\nSummary")
    print("=" * 48)
    print(f"  Tokenizer shards : {config.PARQUET_DIR}/")
    print(f"    {tok_rows:>10,} rows   {tok_chars / 1e6:>8.1f} MB")
    print(f"  Pretrain shards  : {config.PRETRAIN_PARQUET_DIR}/")
    print(f"    {pre_rows:>10,} rows   {pre_chars / 1e6:>8.1f} MB")
    print(f"  Fine-tune shards : {config.FINETUNE_PARQUET_DIR}/")
    print(f"    {ft_rows:>10,} rows   {ft_chars / 1e6:>8.1f} MB")
    total_chars = tok_chars + pre_chars + ft_chars
    print(f"\n  Total written    : {total_chars / 1e6:.1f} MB")
    print("\nDone.")