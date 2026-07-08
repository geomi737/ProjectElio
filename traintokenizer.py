from tokenizers import ByteLevelBPETokenizer
from architecture import eot_token

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["dataset/f_ru_wikipedia.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<unk>", eot_token],
)

tokenizer.save("tokenizer.json", pretty=True)
