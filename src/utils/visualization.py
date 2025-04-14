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

    acc_values = [r["val_acc"] for r in results]
    trial_indices = range(1, len(results) + 1)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(trial_indices, acc_values)
    plt.title(title)
    plt.xlabel("Trial")
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
