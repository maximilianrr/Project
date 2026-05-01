# scripts/download_wtnd.py
# Downloads and cleans the Where There Is No Doctor PDF

import re
import urllib.request
import fitz

# Download
url = "https://ia601902.us.archive.org/24/items/WhereThereIsNoDoctor-English-DavidWerner/14.DavidWerner-WhereThereIsNoDoctor.pdf"
pdf_path = "data/raw/where_there_is_no_doctor.pdf"
out_path = "data/raw/where_there_is_no_doctor_clean.txt"

print("Downloading PDF")
urllib.request.urlretrieve(url, pdf_path)
print(f"Saved to {pdf_path}")

# Extract
print("Extracting text")
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()
print(f"  Extracted {len(text):,} characters from {len(doc)} pages")

# Clean
print("Cleaning")
lines = text.split("\n")
clean_lines = []
for line in lines:
    line = line.strip()
    if not line or len(line) < 20 or line.count(".") > 10:
        continue
    clean_lines.append(line)
clean_text = "\n".join(clean_lines)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(clean_text)

print(f"Saved {len(clean_lines):,} lines to {out_path}")
print("\nDone.")