import torch
import numpy as np
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Import from src
from src.models.model_type import ModelType
from src.utils.save import save_model
from src.utils.data_loader import load_mnist_data
from src.utils.evaluate import test_model
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_training_history,
    plot_random_search_results,
    plot_epoch_comparison,
    plot_hyperparameter_impact,
    plot_decision_boundary,
    plot_tsne_clusters,
    plot_feature_maps,
    plot_prediction_scatter,
    plot_confidence_scatter,
)
from src.optimization.random_search import RandomSearch

from src.experiments.config.mlp_params import mlp_param_space
from src.experiments.config.cnn_params import cnn_param_space

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Create image directory if it doesn't exist
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_dir = f"./images/{timestamp}"
os.makedirs(image_dir, exist_ok=True)

# Parameters
TRAINING_SIZE = 8000
VALIDATION_SIZE = 2000
TEST_SIZE = 2500
BATCH_SIZE = 32
NUM_TRIALS = 5  # Number of trials for random search

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using: {device}")


def main():
    # Load and prepare data
    train_loader, val_loader, test_loader = load_mnist_data(TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE, BATCH_SIZE)

    # Run random search for CNN
    print("\n===== Random Search for CNN =====")
    cnn_random_search = RandomSearch(ModelType.CNN, cnn_param_space, device, num_trials=NUM_TRIALS)
    best_cnn, best_cnn_params, cnn_results = cnn_random_search.search(train_loader, val_loader)
    save_model(best_cnn)

    return 

    # Run random search for MLP
    print("\n===== Random Search for MLP =====")
    mlp_random_search = RandomSearch(ModelType.MLP, mlp_param_space, device, num_trials=NUM_TRIALS)
    best_mlp, best_mlp_params, mlp_results = mlp_random_search.search(train_loader, val_loader)
    save_model(best_mlp)

    # Test the best MLP model
    print("\n===== Evaluation of the best MLP on Test Set =====")
    mlp_accuracy, mlp_cm, mlp_report = test_model(best_mlp, test_loader, device)
    print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
    print("MLP Classification Report:")
    print(mlp_report)
    plot_confusion_matrix(mlp_cm, title="Confusion Matrix - MLP", model_type=ModelType.MLP, save_dir=image_dir)

    # Test the best CNN model
    print("\n===== Evaluation of the best CNN on Test Set =====")
    cnn_accuracy, cnn_cm, cnn_report = test_model(best_cnn, test_loader, device)
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

    # Get numpy arrays from test data for visualization
    X_test_np = []
    y_test_np = []

    # Extract a batch of test data
    test_batch_size = min(1000, TEST_SIZE)  # Limit to 1000 samples for visualization
    test_subset = torch.utils.data.Subset(test_loader.dataset, indices=range(test_batch_size))
    test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=test_batch_size)

    for images, labels in test_subset_loader:
        # Convert to numpy for visualization
        if images.dim() == 4:  # [batch, channels, height, width]
            images_np = images.numpy()
        else:  # [batch, flattened]
            images_np = images.reshape(-1, 28, 28).numpy()

        X_test_np.append(images_np)
        y_test_np.append(labels.numpy())

    X_test_np = np.concatenate(X_test_np)
    y_test_np = np.concatenate(y_test_np)

    # Visualize decision boundaries
    print("\n===== Visualizing Decision Boundaries =====")

    # MLP decision boundaries
    plot_decision_boundary(
        model=best_mlp,
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.MLP,
        reduction_method="tsne",
        title="MLP Decision Boundaries",
        save_dir=image_dir,
        device=device,
    )

    # CNN decision boundaries
    plot_decision_boundary(
        model=best_cnn,
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.CNN,
        reduction_method="tsne",
        title="CNN Decision Boundaries",
        save_dir=image_dir,
        device=device,
    )

    # Visualize t-SNE clusters
    plot_tsne_clusters(
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.MLP,  # Use MLP type for file naming
        title="t-SNE Visualization of MNIST Test Data",
        save_dir=image_dir,
    )

    # Visualize CNN feature maps
    if len(X_test_np) > 0:
        # Select a few sample images for feature map visualization
        for digit in range(min(3, len(X_test_np))):  # Visualize first 3 samples
            sample_image = X_test_np[digit]
            digit_label = y_test_np[digit]

            plot_feature_maps(
                model=best_cnn,
                sample_image=sample_image,
                model_type=ModelType.CNN,
                layer_idx=0,  # First convolutional layer
                title=f"CNN Feature Maps for Digit {digit_label}",
                save_dir=image_dir,
                device=device,
            )

    # Visualize prediction scatter plots
    print("\n===== Creating Prediction Scatter Plots =====")

    # MLP prediction scatter
    plot_prediction_scatter(
        model=best_mlp,
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.MLP,
        reduction_method="tsne",
        title="MLP Predictions on MNIST",
        save_dir=image_dir,
        device=device,
    )

    # CNN prediction scatter
    plot_prediction_scatter(
        model=best_cnn,
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.CNN,
        reduction_method="tsne",
        title="CNN Predictions on MNIST",
        save_dir=image_dir,
        device=device,
    )

    # Visualize confidence scatter plots
    print("\n===== Creating Confidence Scatter Plots =====")

    # MLP confidence scatter
    plot_confidence_scatter(
        model=best_mlp,
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.MLP,
        reduction_method="tsne",
        title="MLP Prediction Confidence",
        save_dir=image_dir,
        device=device,
    )

    # CNN confidence scatter
    plot_confidence_scatter(
        model=best_cnn,
        X_test=X_test_np,
        y_test=y_test_np,
        model_type=ModelType.CNN,
        reduction_method="tsne",
        title="CNN Prediction Confidence",
        save_dir=image_dir,
        device=device,
    )

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
