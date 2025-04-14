import random
import torch.nn as nn


# Definition of the Convolutional neural network (CNN)
class CNN(nn.Module):
    def __init__(
        self,
        num_conv_layers: int,
        num_conv: list[int],
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

        filter = filters[0]

        padding = 1
        stride = 1

        # Create convolutional layers
        for i in range(num_conv_layers):
            if i != 0:
                filter = filter * 2

            conv = random.choice(num_conv)
            print(f"Layer {i}: {conv} convoluzioni - filter {filter}")

            for j in range(conv):
                conv_layers.append(nn.Conv2d(in_channels, filter, kernel_size, padding=padding, stride=stride))

                # Activation function
                match activation_fn:
                    case "relu":
                        conv_layers.append(nn.ReLU())
                    case "softmax":
                        conv_layers.append(nn.Softmax(dim=1))
                    case "sigmoid":
                        conv_layers.append(nn.Sigmoid())

                in_channels = filter

            # Max pooling
            conv_layers.append(nn.MaxPool2d(pool_size))

        self.conv_block = nn.Sequential(*conv_layers)

        # Calculate dimension after convolutional layers
        # For MNIST (28x28) with kernel_size=3, pool_size=2 and padding=1
        size_after_conv = 28
        for i in range(num_conv_layers):
            for i in range(num_conv[0]):
                size_after_conv = ((size_after_conv - kernel_size + 2 * padding) / stride) + 1
            size_after_conv = int(size_after_conv / pool_size)

        # Input dimension of the first fully connected layer
        self.flat_size = filter * (size_after_conv**2)

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
