import torch
from torch.utils.data import DataLoader

from src.models.cnn import CNN
from src.models.mlp import MLP


# Training function with early stopping
def train_model(
    model: CNN | MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    num_epochs=10,
    patience=5,
):
    model = model.to(device)
    best_val_acc = 0.0
    best_model_weights = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Parameters for early stopping
    counter = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping activated at epoch {epoch}")
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluation on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device=device)

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            counter = 0  # Reset early stopping counter
        else:
            counter += 1  # Increment counter

        # Early stopping
        if counter >= patience:
            print(f"No improvement for {patience} consecutive epochs. Early stopping.")
            early_stop = True

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, history


# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy
