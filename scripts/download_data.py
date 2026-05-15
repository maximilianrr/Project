# scripts/download_data.py
# Downloads all raw data needed for the medical chatbot.
#
# Sources:
#   1. MedDialog  (ChatDoctor-HealthCareMagic-100k) — 112k real doctor-patient chats
#   2. MedQuAD    — 16k NIH medical Q&A
#   3. PubMedQA   — 211k biomedical Q&A (long answers only, for medical vocabulary)
#   4. MEDIQA-Chat — clinical doctor-patient dialogues
#   5. Where There Is No Doctor PDF — plain-language health handbook
#
# Run: python scripts/download_data.py

import os
import urllib.request
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

# ── 1. MedDialog ───────────────────────────────────────────────────────────────
print("1/5  Downloading MedDialog (ChatDoctor-HealthCareMagic-100k)...")
ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
ds.save_to_disk("data/raw/meddialog_en")
print(f"     Done. {len(ds):,} examples.")

# ── 2. MedQuAD ────────────────────────────────────────────────────────────────
print("2/5  Cloning MedQuAD...")
os.system("git clone --depth 1 https://github.com/abachaa/MedQuAD.git data/raw/MedQuAD")
print("     Done.")

# # ── 3. PubMedQA ───────────────────────────────────────────────────────────────
# print("3/5  Downloading PubMedQA...")
# pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
# pqa.save_to_disk("data/raw/pubmedqa")
# print(f"     Done. {len(pqa):,} examples.")

# ── 4. MEDIQA-Chat ────────────────────────────────────────────────────────────
print("4/5  Downloading MEDIQA-Chat...")
try:
    mediqa = load_dataset("chiusers/mediqa-chat-2023", split="train")
    mediqa.save_to_disk("data/raw/mediqa_chat")
    print(f"     Done. {len(mediqa):,} examples.")
except Exception as e:
    print(f"     MEDIQA-Chat unavailable ({e}). Skipping.")

# ── 5. Where There Is No Doctor ───────────────────────────────────────────────
print("5/5  Downloading Where There Is No Doctor PDF...")
url = (
    "https://ia601902.us.archive.org/24/items/"
    "WhereThereIsNoDoctor-English-DavidWerner/"
    "14.DavidWerner-WhereThereIsNoDoctor.pdf"
)
urllib.request.urlretrieve(url, "data/raw/where_there_is_no_doctor.pdf")
print("     Done.")

print("\nAll data downloaded.")