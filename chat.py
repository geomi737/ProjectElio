import torch
from transformer import Transformer
from tokenizers import Tokenizer

model_name = "Elio-2.0Instruct" # Убедитесь, что тут имя, под которым сохраняется текущая модель
model_path = f"./models/{model_name}.pth"
tokenizer_path = "tokenizer.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file(tokenizer_path)
eot_token = tokenizer.encode("<eot>").ids[0]

# Загружаем модель
model = Transformer(0.3).to(device)
with open(model_path, "rb") as f:
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model"])
model.eval()

print("🤖 Elio-2.0 Instruct готов к работе! (для выхода напишите 'выход')")

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["выход", "exit", "quit"]:
        break
        
    print("AI: ", end="", flush=True)
    with torch.no_grad():
        # Метод generate теперь сам добавляет eot_token и печатает результат
        model.generate(user_input, 500)
    print()
