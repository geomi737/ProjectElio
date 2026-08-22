import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import tqdm
from traintokenizer import eot_token

# ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
ds = load_dataset("tatsu-lab/alpaca")
data = [dict(row) for row in ds["train"]]

with open("dataset/alpaca.json", "w", encoding="utf-8") as f:
    for l in data:
        f.write(json.dumps(l, ensure_ascii=False) + "\n")
start = True

# with open(raw_output_path, "w") as dresult:
#     for line in tqdm.tqdm(ds["train"]):
#         text = line.get("text")
#         if not start:
#             if text.strip().startswith("=") and text.strip().endswith("=") and text.count("=") == 2:
#                 dresult.write(eot_token + "\n")
#         start = False
#         dresult.write(text)


