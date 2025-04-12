import torch
from src.models.cnn import CNN
from src.models.mlp import MLP


def load_model(path: str, device: torch.device, eval_mode=True):
    # Carica il modello completo
    model: MLP | CNN = torch.load(path, weights_only=False, map_location=device)

    model = model.to(device)

    if eval_mode:
        model.eval()  # Imposta il modello in modalità valutazione (disattiva dropout, etc.)

    print(f"Modello caricato con successo da {path}")
    return model


def inference(model: MLP | CNN, input_data):
    with torch.no_grad():  # Disabilita il calcolo del gradiente durante l'inferenza
        output = model(input_data)

        # Per ottenere la classe predetta (0-9 per MNIST)
        predicted_class = torch.argmax(output, dim=1)

    return predicted_class
