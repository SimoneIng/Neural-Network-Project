import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model_type import ModelType


# Plot confusion matrix
def plot_confusion_matrix(
    cm,
    model_type: ModelType,
    title: str = "Confusion Matrix",
    save_dir="./images",
):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the plot to the images directory with timestamp
    filename = f"{save_dir}/{model_type}_confusion_matrix.png"
    plt.savefig(filename)
    print(f"Saved confusion matrix to {filename}")
    plt.close()


# Plot training history
def plot_training_history(
    history,
    model_type: ModelType,
    title: str = "Training History",
    save_dir="./images",
):
    plt.figure(figsize=(12, 10))

    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{model_type} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title(f"{model_type} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle(title)
    plt.tight_layout()

    # Save the plot to the images directory with timestamp
    filename = f"{save_dir}/{model_type}_training_history.png"
    plt.savefig(filename)
    print(f"Saved training history plot to {filename}")
    plt.close()


# Plot random search results
def plot_random_search_results(
    results,
    title: str,
    model_type: ModelType,
    save_dir="./images",
):
    # Sort results by validation accuracy
    sorted_results = sorted(results, key=lambda x: x["val_acc"], reverse=True)

    acc_values = [r["val_acc"] for r in sorted_results]
    trial_indices = range(1, len(sorted_results) + 1)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(trial_indices, acc_values)
    plt.title(title)
    plt.xlabel("Trial (sorted by accuracy)")
    plt.ylabel("Validation Accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add values above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{acc_values[i]:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the plot to the images directory with timestamp
    filename = f"{save_dir}/{model_type}_random_search_results.png"
    plt.savefig(filename)
    print(f"Saved random search results plot to {filename}")
    plt.close()


# Plot epoch comparison
def plot_epoch_comparison(
    results,
    model_type: ModelType,
    save_dir="./images",
):
    # Group results by number of epochs
    epoch_groups = {}
    for result in results:
        num_epochs = result["params"]["num_epochs"]
        if num_epochs not in epoch_groups:
            epoch_groups[num_epochs] = []
        epoch_groups[num_epochs].append(result["val_acc"])

    # Calculate average accuracy for each epoch group
    epochs = sorted(epoch_groups.keys())
    avg_accuracies = [sum(epoch_groups[e]) / len(epoch_groups[e]) for e in epochs]
    max_accuracies = [max(epoch_groups[e]) for e in epochs]

    # Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(epochs))
    width = 0.35

    plt.bar(x - width / 2, avg_accuracies, width, label="Average Accuracy")
    plt.bar(x + width / 2, max_accuracies, width, label="Max Accuracy")

    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{model_type} - Impact of Number of Epochs")
    plt.xticks(x, epochs)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add values above bars
    for i, v in enumerate(avg_accuracies):
        plt.text(i - width / 2, v + 0.01, f"{v:.4f}", ha="center")
    for i, v in enumerate(max_accuracies):
        plt.text(i + width / 2, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    filename = f"{save_dir}/{model_type}_epoch_comparison.png"
    plt.savefig(filename)
    print(f"Saved epoch comparison plot to {filename}")
    plt.close()


# Additional visualization to show the impact of different hyperparameters
def plot_hyperparameter_impact(
    results,
    param_name,
    model_type: ModelType,
    save_dir="./images",
):
    # Extract unique values for the parameter
    param_values = set()
    for result in results:
        param_values.add(result["params"][param_name])
    param_values = sorted(list(param_values))

    # Group results by parameter value
    param_groups = {val: [] for val in param_values}
    for result in results:
        val = result["params"][param_name]
        param_groups[val].append(result["val_acc"])

    # Calculate statistics
    avg_accuracies = [
        sum(param_groups[val]) / len(param_groups[val]) if param_groups[val] else 0 for val in param_values
    ]
    max_accuracies = [max(param_groups[val]) if param_groups[val] else 0 for val in param_values]

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(param_values))
    width = 0.35

    plt.bar(x - width / 2, avg_accuracies, width, label="Average Accuracy")
    plt.bar(x + width / 2, max_accuracies, width, label="Max Accuracy")

    plt.xlabel(f"{param_name}")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{model_type} - Impact of {param_name}")
    plt.xticks(x, [str(val) for val in param_values])
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add values above bars
    for i, v in enumerate(avg_accuracies):
        plt.text(i - width / 2, v + 0.01, f"{v:.4f}", ha="center")
    for i, v in enumerate(max_accuracies):
        plt.text(i + width / 2, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    filename = f"{save_dir}/{model_type}_{param_name}_impact.png"
    plt.savefig(filename)
    print(f"Saved {param_name} impact plot to {filename}")
    plt.close()
