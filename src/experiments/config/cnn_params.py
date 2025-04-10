# Parameter search space for CNN
cnn_param_space = {
    "num_conv_layers": (1, 3),  # Number of convolutional layers
    "filters": [16, 32, 64, 128],  # Number of filters
    "kernel_size": [3],  # Kernel size (only 3 to avoid errors)
    "pool_size": [2],  # Pool size
    "hidden_layers": (0, 2),  # Number of fully connected layers
    "hidden_size": [64, 128, 256, 512],  # Size of fully connected layers
    "learning_rate": [0.01, 0.001, 0.0001],  # Learning rate
    "optimizer": ["adam", "rprop"],  # Optimizer (ADAM or RProp as requested)
    "activation_fn": ["relu", "sigmoid", "softmax"],  # Activation function
    "num_epochs": [32, 64, 128],  # Number of epochs (added as requested)
}
