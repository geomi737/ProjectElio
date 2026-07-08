import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
import tqdm
from traintokenizer import eot_token

ds = load_dataset("danasone/wikipedia_ru")
raw_output_path = f"dataset/f_ru_wikipedia.txt"

with open(raw_output_path, "w") as dresult:
    for line in tqdm.tqdm(ds["train"]):
        dresult.write(line.get("text") + eot_token)


