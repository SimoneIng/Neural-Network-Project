import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Dataset parameters
TRAINING_SIZE = 8000  # Modified: 8000 for training
VALIDATION_SIZE = 2000  # Modified: 2000 for validation
TEST_SIZE = 2500  # 2500 for testing
BATCH_SIZE = 64

# Load and prepare MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation of MNIST
])

# Load complete dataset
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Select a subset of 10000 examples from the full training dataset
subset_indices = torch.randperm(len(full_dataset))[:TRAINING_SIZE + VALIDATION_SIZE]
dataset_subset = torch.utils.data.Subset(full_dataset, subset_indices)

# Split the subset into training and validation sets
train_dataset, val_dataset = random_split(dataset_subset, [TRAINING_SIZE, VALIDATION_SIZE])

# Select a random subset for the test set
test_indices = torch.randperm(len(test_dataset))[:TEST_SIZE]
test_subset = torch.utils.data.Subset(test_dataset, test_indices)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_subset)}")

# Definition of the Fully Connected neural network (MLP)
class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_sizes, dropout_rate, activation_fn):
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
            elif activation_fn == 'tanh':
                layers.append(nn.Tanh())
            elif activation_fn == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            input_size = hidden_sizes[i]
        
        # Output layer (10 classes)
        layers.append(nn.Linear(input_size, 10))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Definition of the Convolutional neural network (CNN)
class CNN(nn.Module):
    def __init__(self, num_conv_layers, filters, kernel_size, pool_size, hidden_layers, 
                 hidden_sizes, dropout_rate, activation_fn):
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
            if activation_fn == 'relu':
                conv_layers.append(nn.ReLU())
            elif activation_fn == 'tanh':
                conv_layers.append(nn.Tanh())
            elif activation_fn == 'leaky_relu':
                conv_layers.append(nn.LeakyReLU())
            
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
        self.flat_size = filters[-1] * (size_after_conv ** 2)
        
        # Fully connected layers
        fc_layers = [nn.Flatten()]
        input_size = self.flat_size
        
        for i in range(hidden_layers):
            fc_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            
            # Activation function
            if activation_fn == 'relu':
                fc_layers.append(nn.ReLU())
            elif activation_fn == 'tanh':
                fc_layers.append(nn.Tanh())
            elif activation_fn == 'leaky_relu':
                fc_layers.append(nn.LeakyReLU())
            
            # Dropout
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            
            input_size = hidden_sizes[i]
        
        # Output layer
        fc_layers.append(nn.Linear(input_size, 10))
        
        self.fc_block = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

# Training function with early stopping
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, num_epochs=10, patience=5):
    model = model.to(device)
    best_val_acc = 0.0
    best_model_weights = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Parameters for early stopping
    counter = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping activated at epoch {epoch}")
            break
            
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        if scheduler:
            scheduler.step()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Evaluation on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            counter = 0  # Reset early stopping counter
        else:
            counter += 1  # Increment counter
            
        # Early stopping
        if counter >= patience:
            print(f"No improvement for {patience} consecutive epochs. Early stopping.")
            early_stop = True
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model, history

# Evaluation function
def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / total
    accuracy = correct / total
    
    return loss, accuracy

# Function to test the model and get classification report
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, digits=4)
    
    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return accuracy, cm, report

# Plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix', model_type=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the plot to the images directory with timestamp
    filename = f"{image_dir}/{model_type}_confusion_matrix.png"
    plt.savefig(filename)
    print(f"Saved confusion matrix to {filename}")
    plt.close()
# Plot training history
def plot_training_history(history, title='Training History', model_type=None):
    plt.figure(figsize=(12, 10))
    
    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_type} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot to the images directory with timestamp
    filename = f"{image_dir}/{model_type}_training_history.png"
    plt.savefig(filename)
    print(f"Saved training history plot to {filename}")
    plt.close()

# Random Search implementation
class RandomSearch:
    def __init__(self, model_type, param_space, num_trials=10, num_epochs=10):
        self.model_type = model_type
        self.param_space = param_space
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.results = []
    
    def sample_params(self):
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, list):
                params[param_name] = random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int):
                    params[param_name] = random.randint(param_range[0], param_range[1])
                else:
                    params[param_name] = random.uniform(param_range[0], param_range[1])
        
        # Remove weight_decay if optimizer is RProp since it doesn't support it
        if params.get('optimizer') == 'rprop' and 'weight_decay' in params:
            params['weight_decay'] = 0
            
        return params
    
    def create_model(self, params):
        if self.model_type == 'MLP':
            # For MLP
            hidden_layers = params['hidden_layers']
            hidden_sizes = [params['hidden_size']] * hidden_layers
            return MLP(hidden_layers, hidden_sizes, params['dropout_rate'], params['activation_fn'])
        elif self.model_type == 'CNN':
            # For CNN
            num_conv_layers = params['num_conv_layers']
            filters = [params['filters']] * num_conv_layers
            hidden_layers = params['hidden_layers']
            hidden_sizes = [params['hidden_size']] * hidden_layers
            return CNN(num_conv_layers, filters, params['kernel_size'], params['pool_size'],
                       hidden_layers, hidden_sizes, params['dropout_rate'], params['activation_fn'])
    
    def create_optimizer(self, model, params):
        if params['optimizer'] == 'adam':
            return optim.Adam(model.parameters(), lr=params['learning_rate'], 
                             weight_decay=params.get('weight_decay', 0))
        elif params['optimizer'] == 'rprop':  # Added RProp as requested in the project
            # Note: Rprop doesn't support weight_decay parameter
            return optim.Rprop(model.parameters(), lr=params['learning_rate'])
    
    def search(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        best_params = None
        best_model = None
        best_history = None
        
        for i in range(self.num_trials):
            params = self.sample_params()
            print(f"\nTrial {i+1}/{self.num_trials}")
            print(f"Parameters: {params}")
            
            # Create model and optimizer
            model = self.create_model(params)
            optimizer = self.create_optimizer(model, params)
            
            # Train model with early stopping
            model, history = train_model(model, train_loader, val_loader, optimizer, criterion, 
                                         num_epochs=self.num_epochs, patience=20)  # Patience = 3 epochs
            
            # Evaluate on validation set
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            
            # Record results
            self.results.append({
                'params': params,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history
            })
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            # Update best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params
                best_model = model
                best_history = history
                
                # Save loss and accuracy plots for the best model so far
                plot_training_history(history, title=f'Best {self.model_type} Training History (Trial {i+1})', model_type=self.model_type)
        
        # Plot error curves for the best model
        plot_training_history(best_history, title=f'Best {self.model_type} Final Training History', model_type=self.model_type)
        
        return best_model, best_params, self.results

# Parameter search space for MLP
mlp_param_space = {
    'hidden_layers': (1, 4),           # Number of hidden layers
    'hidden_size': [64, 128, 256, 512], # Size of hidden layers
    'dropout_rate': (0.0, 0.5),        # Dropout rate
    'learning_rate': (0.0001, 0.01),   # Learning rate
    'optimizer': ['adam', 'rprop'],    # Optimizer (ADAM or RProp as requested)
    'activation_fn': ['relu', 'tanh', 'leaky_relu'], # Activation function
    'weight_decay': (0.0, 0.001)       # L2 regularization
}

# Parameter search space for CNN
cnn_param_space = {
    'num_conv_layers': (1, 3),          # Number of convolutional layers
    'filters': [16, 32, 64, 128],       # Number of filters
    'kernel_size': [3],                 # Kernel size (only 3 to avoid errors)
    'pool_size': [2],                   # Pool size
    'hidden_layers': (0, 2),            # Number of fully connected layers
    'hidden_size': [64, 128, 256, 512], # Size of fully connected layers
    'dropout_rate': (0.0, 0.5),         # Dropout rate
    'learning_rate': (0.0001, 0.01),    # Learning rate
    'optimizer': ['adam', 'rprop'],     # Optimizer (ADAM or RProp as requested)
    'activation_fn': ['relu', 'tanh', 'leaky_relu'], # Activation function
    'weight_decay': (0.0, 0.001)        # L2 regularization
}

# Number of trials for random search
NUM_TRIALS = 1  # Increased for better exploration
# Number of epochs for training each model
NUM_EPOCHS = 100

# Run random search for MLP
print("\n===== Random Search for MLP =====")
mlp_random_search = RandomSearch('MLP', mlp_param_space, num_trials=NUM_TRIALS, num_epochs=NUM_EPOCHS)
best_mlp, best_mlp_params, mlp_results = mlp_random_search.search(train_loader, val_loader)

# Run random search for CNN
print("\n===== Random Search for CNN =====")
cnn_random_search = RandomSearch('CNN', cnn_param_space, num_trials=NUM_TRIALS, num_epochs=NUM_EPOCHS)
best_cnn, best_cnn_params, cnn_results = cnn_random_search.search(train_loader, val_loader)

# Test the best MLP model
print("\n===== Evaluation of the best MLP on Test Set =====")
mlp_accuracy, mlp_cm, mlp_report = test_model(best_mlp, test_loader)
print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
print("MLP Classification Report:")
print(mlp_report)
plot_confusion_matrix(mlp_cm, title='Confusion Matrix - MLP', model_type="MLP")
plt.savefig('mlp_confusion_matrix.png')
plt.close()

# Test the best CNN model
print("\n===== Evaluation of the best CNN on Test Set =====")
cnn_accuracy, cnn_cm, cnn_report = test_model(best_cnn, test_loader)
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
print("CNN Classification Report:")
print(cnn_report)
plot_confusion_matrix(cnn_cm, title='Confusion Matrix - CNN', model_type="CNN")
plt.savefig('cnn_confusion_matrix.png')
plt.close()

# Compare results
print("\n===== Results Comparison =====")
print(f"MLP Best Parameters: {best_mlp_params}")
print(f"CNN Best Parameters: {best_cnn_params}")
print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

# Plot random search results
def plot_random_search_results(results, title, model_type=None):
    # Sort results by validation accuracy
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    acc_values = [r['val_acc'] for r in sorted_results]
    trial_indices = range(1, len(sorted_results) + 1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(trial_indices, acc_values)
    plt.title(title)
    plt.xlabel('Trial (sorted by accuracy)')
    plt.ylabel('Validation Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc_values[i]:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot to the images directory with timestamp
    filename = f"{image_dir}/{model_type}_random_search_results.png"
    plt.savefig(filename)
    print(f"Saved random search results plot to {filename}")
    plt.close()

plot_random_search_results(mlp_results, 'MLP Random Search Results', model_type="MLP")
plt.savefig('mlp_random_search_results.png')
plt.close()

plot_random_search_results(cnn_results, 'CNN Random Search Results', model_type="CNN")
plt.savefig('cnn_random_search_results.png')
plt.close()

# Final comparison
labels = ['MLP', 'CNN']
test_accuracies = [mlp_accuracy, cnn_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(labels, test_accuracies)
plt.title('Test Set Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values above bars
for i, v in enumerate(test_accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('comparison_results.png')
plt.close()