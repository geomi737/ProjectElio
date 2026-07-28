import torch
from architecture import ModelConfig, Transformer
from tokenizers import Tokenizer
from traintokenizer import eot_token, user_text_token, answer_token

model_name = "Elio-2.0R"
model_path = f"./models/{model_name}/{model_name}.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

modelconfig = ModelConfig(model_name).load_model_layout()
modelconfig.change_generative_params(
    temperature=0.3, top_k=30, top_p=0.85, repeat_penalty=1.0
)
model = Transformer(model_name, modelconfig.get_settings(), device).to(device)
model_type = "R"

tokenizer = model.get_tokenizer()

# Загружаем модель
model.load()
model.eval()

print(f"{model_name} готов к работе! (для выхода напишите 'выход')")

while True:
    temp_input = input("\nUser: ")
    if temp_input.lower() in ["выход", "exit", "quit"]:
        break
    user_input = user_text_token + temp_input + answer_token if model_type == "I" else temp_input

    print("AI: ", end="", flush=True)
    with torch.no_grad():
        model.generate(torch.tensor(tokenizer.encode(user_input).ids), 100)
    print()
