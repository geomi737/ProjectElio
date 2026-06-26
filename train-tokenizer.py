import os
from tokenizers import ByteLevelBPETokenizer

os.chdir("./dataset")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["dataset.txt"], vocab_size=2048, min_frequency=2, special_tokens=["<unk>"]
)

tokenizer.save("tokenizer.json", pretty=True)
