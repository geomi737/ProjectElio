import torch
from architecture import ModelConfig, Transformer
from tokenizers import Tokenizer
from traintokenizer import eot_token, user_text_token, answer_token

model_name = "Elio-1.1I"
model_path = f"./models/{model_name}/{model_name}.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

modelconfig = ModelConfig(model_name).load_model_layout()
modelconfig.change_generative_params(temperature=0.9, top_k=50, top_p=0.9, repeat_penalty=1.2)
model = Transformer(model_name, modelconfig.get_settings(), device).to(device)

tokenizer = model.get_tokenizer()

# Загружаем модель
model.load()
model.eval()

print(f"{model_name}nstruct готов к работе! (для выхода напишите 'выход')")

while True:
    user_input = user_text_token + input("\nUser: ") + answer_token
    if user_input.lower() in ["выход", "exit", "quit"]:
        break
        
    print("AI: ", end="", flush=True)
    with torch.no_grad():
        model.generate(user_input, 100)
    print()
