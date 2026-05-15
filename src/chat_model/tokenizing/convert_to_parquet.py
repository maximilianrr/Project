# src/chat_model/tokenizing/convert_to_parquet.py
"""Convert tokenizer_text.txt and medical JSONL splits to parquet shards.

Produces two output directories:
  data/processed/parquet/          ← tokenizer training shards
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


def _write_shards(rows: list[str], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = [rows[i:i + config.SHARD_SIZE] for i in range(0, len(rows), config.SHARD_SIZE)]
    for i, shard in enumerate(shards):
        out_path = out_dir / f"{prefix}_{i:05d}.parquet"
        table    = pa.table({"text": shard}, schema=TEXT_SCHEMA)
        pq.write_table(table, out_path, compression="snappy")
        tag = "  ← validation shard (nanochat convention)" if i == len(shards) - 1 else ""
        print(f"  {out_path.name:<28} {len(shard):>8,} rows{tag}")


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


def convert_to_parquet() -> None:
    """Convert tokenizer text and fine-tune splits to parquet."""

    # ── 1. Tokenizer parquet ──────────────────────────────────────────────────
    tok_path = Path(config.TOKENIZER_TEXT)
    if not tok_path.exists():
        raise FileNotFoundError(
            f"{tok_path} not found.\n"
            "Run: prepare_tokenizer_data() first."
        )

    print(f"Reading {tok_path}...")
    with tok_path.open(encoding="utf-8") as f:
        rows = [_normalise(line) for line in f if line.strip()]
    print(f"Total lines: {len(rows):,}")

    print(f"\nSaving tokenizer shards to {config.PARQUET_DIR}/")
    _write_shards(rows, Path(config.PARQUET_DIR), "shard")

    # ── 2. Fine-tune parquet (medical splits) ─────────────────────────────────
    finetune_dir = Path(config.FINETUNE_PARQUET_DIR)
    print(f"\nSaving fine-tune shards to {finetune_dir}/")

    for split in ("train", "val", "test"):
        path = Path(config.SPLITS_DIR) / f"{split}.jsonl"
        if not path.exists():
            print(f"  {split}: missing {path} — skipped")
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
            out_path = finetune_dir / f"{split}.parquet"
            table    = pa.table({"text": valid_rows}, schema=TEXT_SCHEMA)
            pq.write_table(table, out_path, compression="snappy")
            print(f"  {out_path.name:<28} {len(valid_rows):>8,} rows")
        else:
            _write_shards(valid_rows, finetune_dir, split)

    print("\nDone.")
    print(f"\nSummary")
    print(f"  Tokenizer shards : {config.PARQUET_DIR}/")
    print(f"  Stage 2 shards   : {config.FINETUNE_PARQUET_DIR}/")


if __name__ == "__main__":
    convert_to_parquet()
