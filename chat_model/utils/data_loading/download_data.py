import os
import urllib.request
import fitz
from datasets import load_dataset
from tqdm import tqdm

def download_data():
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


def download_wtnd(): 
    # Download
    url = "https://ia601902.us.archive.org/24/items/WhereThereIsNoDoctor-English-DavidWerner/14.DavidWerner-WhereThereIsNoDoctor.pdf"
    pdf_path = "data/raw/where_there_is_no_doctor.pdf"
    out_path = "data/raw/where_there_is_no_doctor_clean.txt"

    print("Downloading Where There Is No Doctor PDF")
    
    try: 
        tqdm(urllib.request.urlretrieve(url, pdf_path))
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading WTND: {e}")
        if os.path.exists(pdf_path):
            print("File already exists, proceeding with existing file.")
        else:
            print("Download failed and file does not exist. Please check your internet connection and try again.")
            return
    
    print(f"Saved to {pdf_path}")

    # Extract
    print("Extracting text")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += str(page.get_text("text"))
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