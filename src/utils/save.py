import torch

from src.models.cnn import CNN
from src.models.mlp import MLP


def save_model(model: MLP | CNN):
    file_name = "cnn_model" if isinstance(model, CNN) else "mlp_model"
    path = f"./src/experiments/results/{file_name}.pt"
    
    print(f"SALVATAGGIO: DEVICE {next(model.parameters()).device}")

    # Salva il modello completo (architettura + pesi)
    torch.save(model, path)
    print(f"Modello salvato con successo in {path}")
