from datasets import load_dataset
import tqdm
from transformer import eot_token

ds = load_dataset("danasone/wikipedia_ru")
raw_output_path = "dataset/f_ru_wikipedia.jsonl"

with open(raw_output_path, "w") as dresult:
    for line in tqdm.tqdm(ds["train"]):
        dresult.write(line.get("text") + eot_token)


