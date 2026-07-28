from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
eot_token = "<eot>"
user_text_token = "<user>"
system_text_token = "<system>"
answer_token = "<answer>"

if __name__ == "__main__":
    tokenizer.train(
        files=["dataset/f_ru_wikipedia.txt"],
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["<unk>", eot_token, user_text_token, system_text_token, answer_token],
    )

    tokenizer.save("tokenizer.json", pretty=True)
