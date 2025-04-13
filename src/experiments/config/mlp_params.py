# Parameter search space for MLP
mlp_param_space = {
    "batch_size": [32, 64, 128], 
    "hidden_layers": [1, 2, 4],  # Number of hidden layers
    "hidden_size": [32, 64, 128, 256],  # Size of hidden layers
    "learning_rate": [0.01, 0.001, 0.0001],  # Learning rate
    "optimizer": ["adam", "rprop"],  # Optimizer (ADAM or RProp as requested)
    "activation_fn": ["relu", "sigmoid", "softmax"],  # Activation function
    "num_epochs": [30, 60, 90],  # Number of epochs (added as requested)
}
