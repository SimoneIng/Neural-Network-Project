import torch.nn as nn

# Definition of the Fully Connected neural network (MLP)
class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_sizes, activation_fn):
        super(MLP, self).__init__()
        
        # Input layer (784 = 28x28 pixels)
        layers = [nn.Flatten()]
        input_size = 28 * 28
        
        # Add hidden layers
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_sizes[i]))
            
            # Activation function
            if activation_fn == 'relu':
                layers.append(nn.ReLU())
            elif activation_fn == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif activation_fn == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            input_size = hidden_sizes[i]
        
        # Output layer (10 classes)
        layers.append(nn.Linear(input_size, 10))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)