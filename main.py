import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import from src
from src.models.model_type import ModelType
from src.utils.save import save_model
from src.utils.evaluate import test_model
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_training_history,
    plot_random_search_results,
    plot_epoch_comparison,
    plot_hyperparameter_impact,
)
from src.optimization.random_search import RandomSearch

from src.experiments.config.mlp_params import mlp_param_space
from src.experiments.config.cnn_params import cnn_param_space

from src.utils.constants import NUM_TRIALS


# Create image directory if it doesn't exist
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_dir = f"./images/{timestamp}"
os.makedirs(image_dir, exist_ok=True)


# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using: {device}")


def main():

    # Run random search for CNN
    print("\n===== Random Search for CNN =====")
    cnn_random_search = RandomSearch(ModelType.CNN, cnn_param_space, device, num_trials=NUM_TRIALS)
    best_cnn, best_cnn_params, cnn_results = cnn_random_search.search()
    save_model(best_cnn)

    # Run random search for MLP
    print("\n===== Random Search for MLP =====")
    mlp_random_search = RandomSearch(ModelType.MLP, mlp_param_space, device, num_trials=NUM_TRIALS)
    best_mlp, best_mlp_params, mlp_results = mlp_random_search.search()
    save_model(best_mlp)

    # Test the best MLP model
    print("\n===== Evaluation of the best MLP on Test Set =====")
    mlp_accuracy, mlp_cm, mlp_report = test_model(best_mlp, mlp_random_search.test_ds, device)
    print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
    print("MLP Classification Report:")
    print(mlp_report)
    plot_confusion_matrix(mlp_cm, title="Confusion Matrix - MLP", model_type=ModelType.MLP, save_dir=image_dir)

    # Test the best CNN model
    print("\n===== Evaluation of the best CNN on Test Set =====")
    cnn_accuracy, cnn_cm, cnn_report = test_model(best_cnn, cnn_random_search.test_ds, device)
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
    print("CNN Classification Report:")
    print(cnn_report)
    plot_confusion_matrix(cnn_cm, title="Confusion Matrix - CNN", model_type=ModelType.CNN, save_dir=image_dir)

    # Compare results
    print("\n===== Results Comparison =====")
    print(f"MLP Best Parameters: {best_mlp_params}")
    print(f"CNN Best Parameters: {best_cnn_params}")
    print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

    # Plot training history for best models
    for model_type, history in [(ModelType.MLP, mlp_results[0]["history"]), (ModelType.CNN, cnn_results[0]["history"])]:
        plot_training_history(
            history,
            title=f"Best {model_type} Training History",
            model_type=model_type,
            save_dir=image_dir,
        )

    # Plot random search results
    plot_random_search_results(
        mlp_results,
        title="MLP Random Search Results",
        model_type=ModelType.MLP,
        save_dir=image_dir,
    )
    plot_random_search_results(
        cnn_results,
        title="CNN Random Search Results",
        model_type=ModelType.CNN,
        save_dir=image_dir,
    )

    # Plot epoch comparisons
    plot_epoch_comparison(mlp_results, model_type=ModelType.MLP, save_dir=image_dir)
    plot_epoch_comparison(cnn_results, model_type=ModelType.CNN, save_dir=image_dir)

    # Plot hyperparameter impact for various parameters
    for param in ["learning_rate", "hidden_size", "dropout_rate"]:
        if param in mlp_param_space:
            plot_hyperparameter_impact(mlp_results, param_name=param, model_type=ModelType.MLP, save_dir=image_dir)

    for param in ["learning_rate", "num_filters", "kernel_size", "dropout_rate"]:
        if param in cnn_param_space:
            plot_hyperparameter_impact(cnn_results, param_name=param, model_type=ModelType.CNN, save_dir=image_dir)

    # Final comparison
    labels = [ModelType.MLP.__str__(), ModelType.CNN.__str__()]
    test_accuracies = [mlp_accuracy, cnn_accuracy]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, test_accuracies)
    plt.title("Test Set Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add values above bars
    for i, v in enumerate(test_accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"{image_dir}/comparison_results.png")
    plt.close()

    print("\n===== Visualization Complete =====")
    print(f"All visualizations saved to: {image_dir}")


if __name__ == "__main__":
    main()
