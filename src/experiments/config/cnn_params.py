# Parameter search space for CNN
cnn_param_space = {
    "batch_size": [32, 64, 128],
    "num_conv_layers": [1, 2, 3, 4],  # Number of convolutional layers
    "num_conv": [1, 2, 3],  # number of conv per layer
    "filters": [8, 16, 32],  # Number of filters
    "kernel_size": [3],  # Kernel size (only 3 to avoid errors)
    "pool_size": [2],  # Pool size
    "hidden_layers": [0, 1, 2],  # Number of fully connected layers
    "hidden_size": [32, 64, 128, 256],  # Size of fully connected layers
    "learning_rate": [0.01, 0.001, 0.0001],  # Learning rate
    "optimizer": ["adam", "rprop"],  # Optimizer (ADAM or RProp as requested)
    "activation_fn": ["relu", "sigmoid", "softmax"],  # Activation function
    "num_epochs": [30, 60, 90],  # Number of epochs (added as requested)
}
