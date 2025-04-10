import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# Function to test the model and get classification report
def test_model(model, test_loader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(all_labels, all_preds, digits=4)

    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return accuracy, cm, report
