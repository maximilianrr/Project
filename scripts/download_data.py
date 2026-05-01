# scripts/download_data.py
# Downloads all raw data needed.

import os
import urllib.request
from datasets import load_dataset

# MedDialog 
print("Downloading MedDialog")
ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
ds.save_to_disk("data/raw/meddialog_en")
print(f" Done. {len(ds)} examples saved.")

# MedQuAD 
print("Cloning MedQuAD")
os.system("git clone https://github.com/abachaa/MedQuAD.git data/raw/MedQuAD")
print("Done.")

# Where There Is No Doctor 
print("Downloading Where There Is No Doctor PDF...")
url = "https://ia601902.us.archive.org/24/items/WhereThereIsNoDoctor-English-DavidWerner/14.DavidWerner-WhereThereIsNoDoctor.pdf"
urllib.request.urlretrieve(url, "data/raw/where_there_is_no_doctor.pdf")
print("Done.")

print("\nAll data downloaded.")