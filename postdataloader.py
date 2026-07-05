import numpy as np
from tokenizers import Tokenizer
import tqdm

tokenizer = Tokenizer.from_file("tokenizer.json")
dataset_path = "dataset/f_ru_wikipedia.txt"
binary_output_path = "dataset/f-t-38k_ru_wikipedia.dtst"

with open(binary_output_path, "wb") as dresult, open(dataset_path, "r") as dataset:
    for line in tqdm.tqdm(dataset):
        dresult.write(np.array(tokenizer.encode(line).ids, dtype=np.uint16))


