import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.model_type import ModelType
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


def plot_decision_boundary(
    model,
    X_test,
    y_test,
    model_type: ModelType,
    reduction_method="tsne",
    title="Decision Boundaries",
    save_dir="./images",
    n_samples=1000,  # Limit samples for faster visualization
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Visualize decision boundaries for neural network models on MNIST dataset.

    Args:
        model: Trained PyTorch model
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        model_type: Type of model (CNN or FC)
        reduction_method: 'pca' or 'tsne' for dimensionality reduction
        title: Plot title
        save_dir: Directory to save the plot
        n_samples: Number of test samples to visualize
        device: Device to run the model on
    """
    # Sample a subset of test data for visualization
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_subset = X_test[indices]
    y_subset = y_test[indices]

    # Flatten the images if they're not already flattened
    X_flat = X_subset.reshape(X_subset.shape[0], -1)

    # Apply dimensionality reduction
    if reduction_method.lower() == "pca":
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X_flat)
        reduction_name = "PCA"
    else:  # Default to t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_flat)
        reduction_name = "t-SNE"

    # Create a mesh grid for decision boundary visualization
    h = 0.5  # Step size for the mesh
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # For each point in the mesh, we need to get its corresponding high-dimensional point
    # This is a challenge since dimensionality reduction is not easily invertible
    # We'll use nearest neighbors for approximate inverse mapping

    # Combine mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # For each mesh point, find the nearest neighbor in the reduced space
    mesh_preds = []
    batch_size = 1000  # Process in batches to avoid memory issues

    model.eval()
    with torch.no_grad():
        for i in range(0, len(mesh_points), batch_size):
            batch = mesh_points[i : i + batch_size]
            nn_indices = []

            # Find nearest neighbor for each point in the batch
            for point in batch:
                distances = np.sum((X_2d - point) ** 2, axis=1)
                nn_idx = np.argmin(distances)
                nn_indices.append(nn_idx)

            # Get the corresponding original samples
            orig_samples = X_subset[nn_indices]

            # Prepare for model input
            if model_type == ModelType.CNN:
                # Reshape to [batch_size, channels, height, width]
                orig_samples = orig_samples.reshape(-1, 1, 28, 28)

            # Convert to torch tensor
            tensor_batch = torch.tensor(orig_samples, dtype=torch.float32, device=device)

            # Get predictions
            outputs = model(tensor_batch)
            _, preds = torch.max(outputs, 1)
            mesh_preds.extend(preds.cpu().numpy())

    # Reshape predictions back to mesh shape
    Z = np.array(mesh_preds).reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Spectral")
    plt.colorbar(ticks=range(10), label="Digit Class")

    # Plot the actual points
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_subset, cmap="Spectral", edgecolors="k", s=40, alpha=0.8)

    plt.title(f"{title} ({reduction_name} Projection)")
    plt.xlabel(f"{reduction_name} Component 1")
    plt.ylabel(f"{reduction_name} Component 2")

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Digits")
    plt.gca().add_artist(legend1)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    filename = f"{save_dir}/{model_type}_decision_boundary_{reduction_method}.png"
    plt.savefig(filename)
    print(f"Saved decision boundary plot to {filename}")
    plt.close()


def plot_tsne_clusters(
    X_test,
    y_test,
    model_type: ModelType,
    perplexity=30,
    title="t-SNE Visualization of MNIST",
    save_dir="./images",
    n_samples=2000,
):
    """
    Create a t-SNE visualization of the MNIST data clusters.

    Args:
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        model_type: Type of model (for naming the output file)
        perplexity: t-SNE perplexity parameter
        title: Plot title
        save_dir: Directory to save the plot
        n_samples: Number of test samples to visualize
    """
    # Sample a subset of test data for visualization
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_subset = X_test[indices]
    y_subset = y_test[indices]

    # Flatten the images if they're not already flattened
    X_flat = X_subset.reshape(X_subset.shape[0], -1)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_flat)

    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap="Spectral", edgecolors="k", s=40, alpha=0.8)

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Add color bar
    plt.colorbar(scatter, ticks=range(10), label="Digit Class")

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Digits")
    plt.gca().add_artist(legend1)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save the plot
    filename = f"{save_dir}/{model_type}_tsne_clusters.png"
    plt.savefig(filename)
    print(f"Saved t-SNE clusters plot to {filename}")
    plt.close()


def plot_feature_maps(
    model,
    sample_image,
    model_type: ModelType,
    layer_idx=0,  # Index of the convolutional layer to visualize
    title="Feature Maps Visualization",
    save_dir="./images",
    n_feature_maps=8,  # Number of feature maps to display
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Visualize feature maps from a convolutional neural network.
    Only applicable for CNN models.

    Args:
        model: Trained CNN PyTorch model
        sample_image: A single MNIST image (numpy array)
        model_type: Type of model (should be CNN)
        layer_idx: Index of the convolutional layer to visualize
        title: Plot title
        save_dir: Directory to save the plot
        n_feature_maps: Number of feature maps to display
        device: Device to run the model on
    """
    if model_type != ModelType.CNN:
        print("Feature map visualization is only applicable for CNN models.")
        return

    # Prepare the image
    if len(sample_image.shape) == 2:
        sample_image = sample_image.reshape(1, 1, 28, 28)
    elif len(sample_image.shape) == 3:
        sample_image = sample_image.reshape(1, *sample_image.shape)

    # Convert to torch tensor
    image_tensor = torch.tensor(sample_image, dtype=torch.float32, device=device)

    # Create a hook to get feature maps
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu().numpy())

    # Register the hook
    conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if layer_idx >= len(conv_layers):
        print(f"Layer index {layer_idx} is out of range. The model has {len(conv_layers)} convolutional layers.")
        return

    hook = conv_layers[layer_idx].register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor)

    # Remove the hook
    hook.remove()

    # Get the feature maps
    feature_map = feature_maps[0][0]  # First batch item

    # Plot
    plt.figure(figsize=(15, 10))

    # Plot the original image
    plt.subplot(3, n_feature_maps // 2 + 1, 1)
    plt.imshow(sample_image[0, 0], cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Plot feature maps
    n_display = min(n_feature_maps, feature_map.shape[0])
    for i in range(n_display):
        plt.subplot(3, n_feature_maps // 2 + 1, i + 2)
        plt.imshow(feature_map[i], cmap="viridis")
        plt.title(f"Feature Map {i+1}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    # Save the plot
    filename = f"{save_dir}/{model_type}_feature_maps_layer{layer_idx}.png"
    plt.savefig(filename)
    print(f"Saved feature maps visualization to {filename}")
    plt.close()


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


def plot_prediction_scatter(
    model,
    X_test,
    y_test,
    model_type: ModelType,
    reduction_method="tsne",
    title="Prediction Scatter Plot",
    save_dir="./images",
    n_samples=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Create a scatter plot that shows correct and incorrect predictions by the model.

    Args:
        model: Trained PyTorch model
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        model_type: Type of model (CNN or FC)
        reduction_method: 'pca' or 'tsne' for dimensionality reduction
        title: Plot title
        save_dir: Directory to save the plot
        n_samples: Number of test samples to visualize
        device: Device to run the model on
    """
    # Sample a subset of test data for visualization
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_subset = X_test[indices]
    y_subset = y_test[indices]

    # Flatten the images if they're not already flattened
    X_flat = X_subset.reshape(X_subset.shape[0], -1)

    # Apply dimensionality reduction
    if reduction_method.lower() == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X_flat)
        reduction_name = "PCA"
    else:  # Default to t-SNE
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_flat)
        reduction_name = "t-SNE"

    # Prepare model input
    if model_type == ModelType.CNN:
        # Reshape for CNN input [batch, channels, height, width]
        model_input = torch.tensor(X_subset.reshape(-1, 1, 28, 28), dtype=torch.float32, device=device)
    else:
        # Flatten for MLP input
        model_input = torch.tensor(X_subset.reshape(-1, 784), dtype=torch.float32, device=device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(model_input)
        _, predictions = torch.max(outputs, 1)

    # Convert predictions to numpy
    predictions = predictions.cpu().numpy()

    # Create arrays for correct and incorrect predictions
    correct_mask = predictions == y_subset
    incorrect_mask = ~correct_mask

    # Create the scatter plot
    plt.figure(figsize=(12, 10))

    # Plot correct predictions
    plt.scatter(
        X_2d[correct_mask, 0],
        X_2d[correct_mask, 1],
        c=y_subset[correct_mask],
        marker="o",
        cmap="Spectral",
        alpha=0.7,
        s=50,
        edgecolors="w",
        label="Correct Prediction",
    )

    # Plot incorrect predictions
    plt.scatter(
        X_2d[incorrect_mask, 0],
        X_2d[incorrect_mask, 1],
        c=y_subset[incorrect_mask],
        marker="X",
        cmap="Spectral",
        alpha=1.0,
        s=80,
        edgecolors="black",
        linewidth=1.5,
        label="Incorrect Prediction",
    )

    # Add colorbar
    cbar = plt.colorbar(label="Digit Class")
    cbar.set_ticks(range(10))

    # Calculate accuracy for the subset
    accuracy = correct_mask.sum() / len(y_subset)

    plt.title(f"{title} - Accuracy: {accuracy:.4f}\n({reduction_name} Projection)")
    plt.xlabel(f"{reduction_name} Component 1")
    plt.ylabel(f"{reduction_name} Component 2")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.3)

    # Add some annotations for misclassified points
    if np.any(incorrect_mask):
        # Find indices of some misclassified points to annotate
        incorrect_indices = np.where(incorrect_mask)[0]
        max_annotations = min(10, len(incorrect_indices))  # Limit to 10 annotations
        annotation_indices = np.random.choice(incorrect_indices, max_annotations, replace=False)

        for idx in annotation_indices:
            plt.annotate(
                f"True: {y_subset[idx]}, Pred: {predictions[idx]}",
                (X_2d[idx, 0], X_2d[idx, 1]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    plt.tight_layout()

    # Save the plot
    filename = f"{save_dir}/{model_type}_prediction_scatter.png"
    plt.savefig(filename)
    print(f"Saved prediction scatter plot to {filename}")
    plt.close()


def plot_confidence_scatter(
    model,
    X_test,
    y_test,
    model_type: ModelType,
    reduction_method="tsne",
    title="Prediction Confidence Scatter Plot",
    save_dir="./images",
    n_samples=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Create a scatter plot showing prediction confidence levels.

    Args:
        model: Trained PyTorch model
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        model_type: Type of model (CNN or FC)
        reduction_method: 'pca' or 'tsne' for dimensionality reduction
        title: Plot title
        save_dir: Directory to save the plot
        n_samples: Number of test samples to visualize
        device: Device to run the model on
    """
    # Sample a subset of test data for visualization
    indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_subset = X_test[indices]
    y_subset = y_test[indices]

    # Flatten the images if they're not already flattened
    X_flat = X_subset.reshape(X_subset.shape[0], -1)

    # Apply dimensionality reduction
    if reduction_method.lower() == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X_flat)
        reduction_name = "PCA"
    else:  # Default to t-SNE
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_flat)
        reduction_name = "t-SNE"

    # Prepare model input
    if model_type == ModelType.CNN:
        # Reshape for CNN input [batch, channels, height, width]
        model_input = torch.tensor(X_subset.reshape(-1, 1, 28, 28), dtype=torch.float32, device=device)
    else:
        # Flatten for MLP input
        model_input = torch.tensor(X_subset.reshape(-1, 784), dtype=torch.float32, device=device)

    # Get model predictions with probabilities
    model.eval()
    with torch.no_grad():
        outputs = model(model_input)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

    # Convert to numpy
    predictions = predictions.cpu().numpy()
    confidences = confidences.cpu().numpy()

    # Create the scatter plot
    plt.figure(figsize=(12, 10))

    # Create a scatter plot with confidence values as color
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=confidences, cmap="viridis", s=60, alpha=0.8, edgecolors="k")

    # Mark incorrect predictions
    incorrect_mask = predictions != y_subset
    if np.any(incorrect_mask):
        plt.scatter(
            X_2d[incorrect_mask, 0],
            X_2d[incorrect_mask, 1],
            s=100,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            marker="o",
            label="Incorrect Prediction",
        )

    # Add colorbar for confidence values
    cbar = plt.colorbar(scatter, label="Prediction Confidence")
    cbar.set_ticks(np.linspace(0, 1, 11))

    # Calculate accuracy for the subset
    accuracy = (predictions == y_subset).sum() / len(y_subset)

    plt.title(f"{title} - Accuracy: {accuracy:.4f}\n({reduction_name} Projection)")
    plt.xlabel(f"{reduction_name} Component 1")
    plt.ylabel(f"{reduction_name} Component 2")

    # Add legend if we have incorrect predictions
    if np.any(incorrect_mask):
        plt.legend(loc="upper right")

    plt.grid(True, linestyle="--", alpha=0.3)

    # Add annotations for some low confidence correct predictions and high confidence incorrect predictions
    high_conf_incorrect = np.where((incorrect_mask) & (confidences > 0.8))[0]
    low_conf_correct = np.where((~incorrect_mask) & (confidences < 0.6))[0]

    # Annotate high confidence incorrect predictions
    for idx in high_conf_incorrect[:5]:  # Limit to 5 annotations
        plt.annotate(
            f"True: {y_subset[idx]}, Pred: {predictions[idx]}, Conf: {confidences[idx]:.2f}",
            (X_2d[idx, 0], X_2d[idx, 1]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    # Annotate low confidence correct predictions
    for idx in low_conf_correct[:5]:  # Limit to 5 annotations
        plt.annotate(
            f"True: {y_subset[idx]}, Pred: {predictions[idx]}, Conf: {confidences[idx]:.2f}",
            (X_2d[idx, 0], X_2d[idx, 1]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.tight_layout()

    # Save the plot
    filename = f"{save_dir}/{model_type}_confidence_scatter.png"
    plt.savefig(filename)
    print(f"Saved confidence scatter plot to {filename}")
    plt.close()
