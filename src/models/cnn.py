import torch.nn as nn


# Definition of the Convolutional neural network (CNN)
class CNN(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        filters: list,
        kernel_size: int,
        pool_size: int,
        hidden_layers: int,
        hidden_sizes: list,
        activation_fn: str,
    ):
        super(CNN, self).__init__()

        conv_layers = []
        in_channels = 1  # MNIST has one channel (grayscale)

        # Make sure kernel_size is not too large
        # Maximum kernel_size depends on the number of convolutional layers
        if num_conv_layers > 1 and kernel_size > 3:
            kernel_size = 3  # Limit kernel_size to avoid dimension issues

        # Create convolutional layers
        for i in range(num_conv_layers):
            # Add padding=1 to maintain image dimensions
            conv_layers.append(nn.Conv2d(in_channels, filters[i], kernel_size, padding=1))

            # Activation function
            match activation_fn:
                case "relu":
                    conv_layers.append(nn.ReLU())
                case "softmax":
                    conv_layers.append(nn.Softmax(dim=1))
                case "sigmoid":
                    conv_layers.append(nn.Sigmoid())

            # Max pooling
            conv_layers.append(nn.MaxPool2d(pool_size))

            in_channels = filters[i]

        self.conv_block = nn.Sequential(*conv_layers)

        # Calculate dimension after convolutional layers
        # For MNIST (28x28) with kernel_size=3, pool_size=2 and padding=1
        size_after_conv = 28
        for i in range(num_conv_layers):
            # Convolution with padding=1: output = (input - kernel_size + 2*padding + 1) = input
            # size_after_conv remains unchanged thanks to padding
            # Max pooling: output = input / pool_size
            size_after_conv = size_after_conv // pool_size

        # Input dimension of the first fully connected layer
        self.flat_size = filters[-1] * (size_after_conv**2)

        # Fully connected layers
        fc_layers = [nn.Flatten()]
        input_size = self.flat_size

        for i in range(hidden_layers):
            fc_layers.append(nn.Linear(input_size, hidden_sizes[i]))

            # Activation function
            match activation_fn:
                case "relu":
                    conv_layers.append(nn.ReLU())
                case "softmax":
                    conv_layers.append(nn.Softmax(dim=1))
                case "sigmoid":
                    conv_layers.append(nn.Sigmoid())

            input_size = hidden_sizes[i]

        # Output layer
        fc_layers.append(nn.Linear(input_size, 10))

        self.fc_block = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
