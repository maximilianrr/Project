#scripts/preprocess.py
"""
Preprocesses raw data into nanochat conversation format.
Output: data/splits/train.jsonl, val.jsonl, test.jsonl
Run: python scripts/preprocess.py
"""
import os
import json
import glob
import random
import re
import xml.etree.ElementTree as ET
from datasets import load_from_disk

random.seed(42)

def clean(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces/newlines with a single space
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text) # Remove bracketed content
    return text.strip() 

def to_conversation(q, a):
    return [
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a}
    ] #Nanochat expects such format

def load_medquad(path):
    pairs = []
    for xml_file in glob.glob(os.path.join(path, "**/*.xml"), recursive=True):
        root = ET.parse(xml_file).getroot()
        for qa in root.findall(".//QAPair"):
            q = qa.find("Question")
            a = qa.find("Answer")
            if a is not None and a.text and len(a.text.strip()) > 20: # Filter out missing answers and very short answers
                q_text = clean(q.text or "")
                a_text = clean(a.text or "")
                if q_text and a_text:
                    pairs.append(to_conversation(q_text, a_text)) # Convert to chat format
    print(f"  MedQuAD: {len(pairs)} examples")
    return pairs

def load_meddialog(path):
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        q = item["input"].strip()
        a = item["output"].strip()
        if q and a:
            pairs.append(to_conversation(q, a))
    print(f"  MedDialog: {len(pairs)} examples")
    return pairs

def save_splits(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(data)
    n = len(data)
    train = data[:int(n * 0.85)] # 85%
    val   = data[int(n * 0.85):int(n * 0.95)] #10%
    test  = data[int(n * 0.95):] # 5%
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(out_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item) + "\n")
        print(f"  {name}: {len(split)} examples → {out_path}")

print("Loading data")
data = load_medquad("data/raw/MedQuAD") + load_meddialog("data/raw/meddialog_en")
print(f"Total: {len(data)} examples")

print("Saving splits")
save_splits(data, "data/splits")
print("\nDone.")