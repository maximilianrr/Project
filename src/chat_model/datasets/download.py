# src/chat_model/datasets/download.py
"""Download all raw data sources for both training stages.

Stage 1 (general pretraining):
  - karpathy/climbmix-400b-shuffle  — raw general English text  → ChunkTextDataset
  - OpenAssistant/oasst2            — conversational Q/A pairs  → ChunkChatDataset

Stage 2 (medical fine-tuning):
  - MedDialog (ChatDoctor-HealthCareMagic) — real patient/doctor conversations
  - MedQuAD                               — NIH factual medical Q&A
  - MEDIQA-Chat                           — clinical dialogues (optional)
  - PubMed abstracts                      — dense medical vocabulary exposure
  - WTND book                             — plain-language medical prose
"""

from __future__ import annotations

import ftplib
import gzip
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from chat_model import config


# ── Shared text cleaning ───────────────────────────────────────────────────────

_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{4,}")
_URL       = re.compile(r"https?://\S+|www\.\S+")


def _clean(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n\n", text)
    return text.strip()


# ── Stage 1 — climbmix raw text ───────────────────────────────────────────────

def download_climbmix(max_documents: int | None = None) -> None:
    """
    Downloads karpathy/climbmix-400b-shuffle and saves raw text documents as
    parquet shards for Stage 1 pretraining with ChunkTextDataset.

    This is the primary Stage 1 source: clean general English text with no
    medical bias, giving the model solid language foundations before fine-tuning.
    """
    out_dir = Path(config.CLIMBMIX_PARQUET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("climbmix_*.parquet"))
    if existing:
        print(f"  climbmix: found {len(existing)} existing shard(s) — skipping download.")
        return

    print("Downloading karpathy/climbmix-400b-shuffle...")
    print("  (this is a large dataset — use max_documents to cap for testing)")

    ds = load_dataset(
        "karpathy/climbmix-400b-shuffle",
        split="train",
        streaming=True,   # stream so we don't have to materialise 400 B tokens
    )

    _CLIMBMIX_SCHEMA = pa.schema([pa.field("text", pa.string())])

    records = []
    shard_idx = 0
    total_chars = 0
    count = 0

    for row in ds:
        text = _clean(row.get("text") or "")
        if not text or len(text) < 50:
            continue

        records.append({"text": text})
        total_chars += len(text)
        count += 1

        if len(records) % 10_000 == 0:
            print(f"  Collected {count:,} documents ({total_chars / 1e9:.2f} GB)...")

        # Flush a shard
        if len(records) >= config.SHARD_SIZE:
            out_path = out_dir / f"climbmix_{shard_idx:05d}.parquet"
            table = pa.table({"text": [r["text"] for r in records]}, schema=_CLIMBMIX_SCHEMA)
            pq.write_table(table, out_path, compression="snappy")
            shard_chars = sum(len(r["text"]) for r in records)
            print(f"  climbmix_{shard_idx:05d}.parquet  {len(records):,} docs  {shard_chars / 1e6:>6.1f} MB")
            records = []
            shard_idx += 1

        if max_documents and count >= max_documents:
            break

    # Flush remaining
    if records:
        out_path = out_dir / f"climbmix_{shard_idx:05d}.parquet"
        table = pa.table({"text": [r["text"] for r in records]}, schema=_CLIMBMIX_SCHEMA)
        pq.write_table(table, out_path, compression="snappy")
        shard_chars = sum(len(r["text"]) for r in records)
        print(f"  climbmix_{shard_idx:05d}.parquet  {len(records):,} docs  {shard_chars / 1e6:>6.1f} MB")

    print(f"  Total documents: {count:,}  |  ~{total_chars / 1e9:.2f} GB text")
    print(f"  Done. climbmix saved to {out_dir}/")


# ── Stage 1 — oasst2 conversational data ──────────────────────────────────────

LANG        = "en"
MIN_Q_CHARS = 20
MIN_A_CHARS = 40
MAX_Q_CHARS = 2000
MAX_A_CHARS = 3000


def _is_usable_pair(question: str, answer: str) -> bool:
    if not (MIN_Q_CHARS <= len(question) <= MAX_Q_CHARS):
        return False
    if not (MIN_A_CHARS <= len(answer) <= MAX_A_CHARS):
        return False
    if question.lower() == answer.lower():
        return False
    return True


def download_pretrain_data(max_pairs: int | None = None) -> None:
    """
    Downloads OpenAssistant oasst2 and saves as parquet shards for Stage 1.
    Reconstructs Q/A pairs from the message tree using parent_id.
    English only, high-quality messages only.
    Used alongside climbmix in Stage 1 — loaded with ChunkChatDataset.
    """
    out_dir = Path(config.PRETRAIN_PARQUET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("oasst2_*.parquet"))
    if existing:
        print(f"  oasst2: found {len(existing)} existing shard(s) — skipping download.")
        return

    print("Downloading OpenAssistant/oasst2...")
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    print(f"  Total messages: {len(ds):,}")

    msg_index = {row["message_id"]: row for row in ds}

    pairs = []
    skipped = {
        "lang": 0,
        "quality": 0,
        "deleted": 0,
        "length": 0,
        "no_parent": 0,
        "non_prompter_parent": 0,
    }

    for msg in ds:
        if msg.get("role") != "assistant":
            continue
        if msg.get("lang") != LANG:
            skipped["lang"] += 1
            continue
        if msg.get("deleted", False):
            skipped["deleted"] += 1
            continue
        if not msg.get("review_result", True):
            skipped["quality"] += 1
            continue

        parent_id = msg.get("parent_id")
        parent    = msg_index.get(parent_id) if parent_id else None
        if parent is None:
            skipped["no_parent"] += 1
            continue
        if parent.get("role") != "prompter":
            skipped["non_prompter_parent"] += 1
            continue
        if parent.get("lang") != LANG or parent.get("deleted", False):
            skipped["lang"] += 1
            continue

        question = _clean(parent.get("text") or "")
        answer   = _clean(msg.get("text") or "")

        if not _is_usable_pair(question, answer):
            skipped["length"] += 1
            continue

        pairs.append({"question": question, "answer": answer})
        if max_pairs and len(pairs) >= max_pairs:
            break

    print(f"  Pairs extracted: {len(pairs):,}")
    for label, count in skipped.items():
        if count:
            print(f"  Skipped {label}: {count:,}")

    schema = pa.schema([pa.field("question", pa.string()), pa.field("answer", pa.string())])
    shards = [pairs[i:i + config.SHARD_SIZE] for i in range(0, len(pairs), config.SHARD_SIZE)]

    total_chars = 0
    for i, shard in enumerate(shards):
        out_path = out_dir / f"oasst2_{i:05d}.parquet"
        table    = pa.table(
            {"question": [p["question"] for p in shard],
             "answer":   [p["answer"]   for p in shard]},
            schema=schema,
        )
        pq.write_table(table, out_path, compression="snappy")

        shard_chars = sum(len(p["question"]) + len(p["answer"]) for p in shard)
        total_chars += shard_chars
        print(f"  oasst2_{i:05d}.parquet  {len(shard):,} pairs  {shard_chars / 1e6:>6.1f} MB")

    print(f"  Total text size: ~{total_chars / 1e6:.0f} MB")
    print(f"  Done. {len(pairs):,} oasst2 pairs saved to {out_dir}/")


# ── Stage 2 — PubMed abstracts ────────────────────────────────────────────────

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_DIR  = "/pubmed/baseline"

MIN_ABSTRACT_CHARS = 100
MAX_ABSTRACT_CHARS = 4000

_PUBMED_SCHEMA = pa.schema([
    pa.field("title",    pa.string()),
    pa.field("abstract", pa.string()),
])


def _iter_ftp_abstracts(max_abstracts: int | None):
    """
    Connect to NCBI FTP, iterate every .xml.gz file in the baseline directory,
    and yield dicts with 'title' and 'abstract' keys.
    """
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(FTP_DIR)

    gz_files = sorted(f for f in ftp.nlst() if f.endswith(".xml.gz"))
    print(f"  Found {len(gz_files)} XML.gz files on NCBI FTP baseline.")

    count = 0
    for filename in gz_files:
        print(f"  Fetching {filename} ...", flush=True)
        buf = BytesIO()
        ftp.retrbinary(f"RETR {filename}", buf.write)
        buf.seek(0)

        with gzip.open(buf, "rb") as f:
            tree = ET.parse(f)

        for article in tree.findall(".//PubmedArticle"):
            title_el = article.find(".//ArticleTitle")
            title    = _clean(title_el.text or "" if title_el is not None else "")

            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (el.text or "") for el in abstract_parts if el.text
            ) if abstract_parts else ""
            abstract = _clean(abstract)[:MAX_ABSTRACT_CHARS]

            if not (MIN_ABSTRACT_CHARS <= len(abstract) <= MAX_ABSTRACT_CHARS):
                continue

            yield {"title": title, "abstract": abstract}
            count += 1

            if max_abstracts and count >= max_abstracts:
                ftp.quit()
                return

    ftp.quit()


def download_pubmed(max_abstracts: int | None = None) -> None:
    """
    Downloads PubMed abstracts from NCBI FTP baseline and saves as parquet
    shards for Stage 2 fine-tuning. Gives the model exposure to real medical
    vocabulary before/during fine-tuning on conversational medical data.

    NOTE: PubMed is Stage 2 only — it is NOT included in Stage 1 pretraining
    so as not to bias the model's general English language foundations.
    """
    out_dir = Path(config.PUBMED_PARQUET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("pubmed_*.parquet"))
    if existing:
        print(f"  PubMed: found {len(existing)} existing shard(s) — skipping download.")
        return

    print(f"Connecting to {FTP_HOST} ...")
    print("  This can take several minutes for the full dataset.")

    records = []
    for record in _iter_ftp_abstracts(max_abstracts):
        records.append(record)
        if len(records) % 100_000 == 0:
            print(f"  Collected {len(records):,} abstracts...")

    print(f"  Abstracts extracted: {len(records):,}")

    shards = [records[i:i + config.SHARD_SIZE] for i in range(0, len(records), config.SHARD_SIZE)]
    total_chars = 0

    for i, shard in enumerate(shards):
        out_path = out_dir / f"pubmed_{i:05d}.parquet"
        table    = pa.table(
            {"title":    [r["title"]    for r in shard],
             "abstract": [r["abstract"] for r in shard]},
            schema=_PUBMED_SCHEMA,
        )
        pq.write_table(table, out_path, compression="snappy")

        shard_chars = sum(len(r["title"]) + len(r["abstract"]) for r in shard)
        total_chars += shard_chars
        print(f"  pubmed_{i:05d}.parquet  {len(shard):,} rows  {shard_chars / 1e6:>6.1f} MB")

    print(f"  Total text size: ~{total_chars / 1e6:.0f} MB")
    print(f"  Done. {len(records):,} PubMed abstracts saved to {out_dir}/")


# ── Stage 2 — medical conversational sources ──────────────────────────────────

def download_meddialog() -> None:
    path = Path(config.MEDDIALOG_DIR)
    if path.exists():
        print(f"  MedDialog: already exists — skipping.")
        return
    print("  Downloading MedDialog (ChatDoctor-HealthCareMagic-100k)...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    ds.save_to_disk(str(path))
    print(f"  Done. {len(ds):,} examples.")


def download_medquad() -> None:
    path = Path(config.MEDQUAD_DIR)
    if path.exists():
        print(f"  MedQuAD: already exists — skipping.")
        return
    if not shutil.which("git"):
        raise RuntimeError("git is required to clone MedQuAD.")
    print("  Cloning MedQuAD...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/abachaa/MedQuAD.git", str(path)],
        check=True,
    )
    print(f"  Done.")


def download_mediqa() -> None:
    path = Path(config.MEDIQA_DIR)
    if path.exists():
        print(f"  MEDIQA-Chat: already exists — skipping.")
        return
    try:
        print("  Downloading MEDIQA-Chat (optional)...")
        ds = load_dataset("chiusers/mediqa-chat-2023", split="train")
        ds.save_to_disk(str(path))
        print(f"  Done. {len(ds):,} examples.")
    except Exception as exc:
        print(f"  MEDIQA-Chat unavailable — skipping. ({exc})")


def download_wtnd() -> None:
    """Downloads the WTND PDF, extracts and cleans text for tokenizer vocabulary."""
    pdf_path = Path(config.WTND_PDF)
    out_path = Path(config.WTND_CLEAN)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print("  Downloading Where There Is No Doctor PDF...")
        url = (
            "https://ia601902.us.archive.org/24/items/"
            "WhereThereIsNoDoctor-English-DavidWerner/"
            "14.DavidWerner-WhereThereIsNoDoctor.pdf"
        )
        urllib.request.urlretrieve(url, str(pdf_path))
        print(f"  Saved to {pdf_path}")
    else:
        print(f"  WTND PDF: already exists — skipping download.")

    print("  Extracting and cleaning WTND text...")
    with fitz.open(str(pdf_path)) as doc:
        raw_text = "\n".join(page.get_text() for page in doc)
        print(f"  Extracted {len(raw_text):,} chars from {len(doc)} pages")

    clean_lines = []
    pending = ""
    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) < 25:
            continue
        if re.fullmatch(r"[\d\s.\-]+", line):
            continue
        if sum(c.isalpha() for c in line) / max(len(line), 1) < 0.60:
            continue
        if pending:
            line = pending + line
            pending = ""
        if line.endswith("-"):
            pending = line[:-1]
            continue
        clean_lines.append(line)

    clean_text = re.sub(r"\n{3,}", "\n\n", "\n".join(clean_lines)).strip()
    out_path.write_text(clean_text, encoding="utf-8")
    print(f"  Saved {clean_text.count(chr(10)) + 1:,} lines to {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def download_data(
    max_documents:  int | None = None,
    max_pairs:      int | None = None,
    max_abstracts:  int | None = None,
) -> None:
    """Downloads all data sources for Stage 1 and Stage 2.

    Stage 1 (pretraining):
      - climbmix  : raw general English text  → ChunkTextDataset
      - oasst2    : English Q/A conversations → ChunkChatDataset

    Stage 2 (medical fine-tuning):
      - MedDialog, MedQuAD, MEDIQA-Chat, PubMed, WTND
    """
    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)

    print("\nStage 1 — Pretraining data")
    print("=" * 48)
    download_climbmix(max_documents=max_documents)
    download_pretrain_data(max_pairs=max_pairs)

    print("\nStage 2 — Medical fine-tuning data")
    print("=" * 48)
    download_meddialog()
    download_medquad()
    download_mediqa()
    download_pubmed(max_abstracts=max_abstracts)
    download_wtnd()

    print("\nAll data downloaded.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download all raw data sources.")
    parser.add_argument("--max-documents", type=int, default=None,
                        help="Cap on climbmix documents (e.g. 50000 for a quick test).")
    parser.add_argument("--max-pairs",     type=int, default=None,
                        help="Cap on oasst2 pairs (e.g. 1000 for a quick test).")
    parser.add_argument("--max-abstracts", type=int, default=None,
                        help="Cap on PubMed abstracts (e.g. 50000 for a quick test).")
    args = parser.parse_args()
    download_data(
        max_documents=args.max_documents,
        max_pairs=args.max_pairs,
        max_abstracts=args.max_abstracts,
    )