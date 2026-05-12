import os
import urllib.request
import fitz
from datasets import load_dataset
from tqdm import tqdm
import sys

# Add src to path if needed (allows running this script directly)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(_current_dir, "..", "..")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from chat_model.datasets.preprocess import preprocess
from chat_model import config

def download_data():
    """
    Downloads the raw data files for MedDialog, MedQuAD, and Where There Is No Doctor.
    Saves them to the RAW_DIR specified in config.py. For MedDialog, saves the dataset in Hugging Face format. For MedQuAD, clones the GitHub repository. For WTND, downloads the PDF and extracts the text.
    """

    os.makedirs(config.RAW_DIR, exist_ok=True)
    # MedDialog 
    print("Downloading MedDialog")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    ds.save_to_disk(config.MEDDIALOG_DIR)
    print(f" Done. {len(ds)} examples saved.")

    # MedQuAD 
    print("Cloning MedQuAD")
    if os.path.exists(config.MEDQUAD_DIR):
        print(f"Directory {config.MEDQUAD_DIR} already exists, skipping clone.")
    else:
        os.system(f"git clone https://github.com/abachaa/MedQuAD.git {config.MEDQUAD_DIR}")
    print("Done.")

    # Where There Is No Doctor 
    download_wtnd()

    print("\nAll data downloaded.")


def download_wtnd(): 
    """
    Downloads the Where There Is No Doctor PDF and extracts the text.
    """

    # Download
    url = "https://ia601902.us.archive.org/24/items/WhereThereIsNoDoctor-English-DavidWerner/14.DavidWerner-WhereThereIsNoDoctor.pdf"
    pdf_path = config.WTND_PDF
    out_path = config.WTND_CLEAN

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


if __name__ == "__main__":
    download_data()
    preprocess()
