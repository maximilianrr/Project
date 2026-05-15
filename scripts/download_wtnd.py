# scripts/download_wtnd.py
# Downloads and cleans the "Where There Is No Doctor" PDF.
# Produces a clean plain-text version for tokenizer training and preprocessing.
#
# Run: python scripts/download_wtnd.py

import re
import urllib.request
import fitz  # PyMuPDF

PDF_URL  = (
    "https://ia601902.us.archive.org/24/items/"
    "WhereThereIsNoDoctor-English-DavidWerner/"
    "14.DavidWerner-WhereThereIsNoDoctor.pdf"
)
PDF_PATH = "data/raw/where_there_is_no_doctor.pdf"
OUT_PATH = "data/raw/where_there_is_no_doctor_clean.txt"

import os
os.makedirs("data/raw", exist_ok=True)

# ── Download ───────────────────────────────────────────────────────────────────
print("Downloading PDF...")
urllib.request.urlretrieve(PDF_URL, PDF_PATH)
print(f"  Saved to {PDF_PATH}")

# ── Extract ────────────────────────────────────────────────────────────────────
print("Extracting text...")
doc = fitz.open(PDF_PATH)
pages_text = []
for page in doc:
    pages_text.append(page.get_text())
raw_text = "\n".join(pages_text)
print(f"  Extracted {len(raw_text):,} characters from {len(doc)} pages")

# ── Clean ──────────────────────────────────────────────────────────────────────
print("Cleaning...")

def clean_wtnd(text: str) -> str:
    lines = text.split("\n")
    clean = []
    for line in lines:
        line = line.strip()

        # Skip very short lines (likely headers, page numbers, figure captions)
        if len(line) < 25:
            continue

        # Skip lines that look like page numbers or figure references
        if re.fullmatch(r'[\d\s\.\-]+', line):
            continue

        # Skip lines that are mostly non-alphabetic (table borders, etc.)
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.60:
            continue

        # Merge hyphenated line-breaks (e.g. "treat-\nment" → "treatment")
        line = re.sub(r'-\s*$', '', line)

        clean.append(line)

    # Re-join, collapsing multiple blank lines
    joined = "\n".join(clean)
    joined = re.sub(r'\n{3,}', '\n\n', joined)
    return joined.strip()

clean_text = clean_wtnd(raw_text)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(clean_text)

line_count = clean_text.count("\n") + 1
print(f"  Saved {line_count:,} lines to {OUT_PATH}")
print("Done.")