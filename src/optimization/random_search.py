import torch.nn as nn
import torch.optim as optim
import random

from src.models.cnn import CNN
from src.models.mlp import MLP
from src.utils.train import evaluate_model, train_model
from src.utils.visualization import plot_training_history


# Random Search implementation
class RandomSearch:
    def __init__(self, model_type, param_space, device, num_trials=10):
        self.model_type = model_type
        self.param_space = param_space
        self.num_trials = num_trials
        self.device = device
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
            
        return params
    
    def create_model(self, params):
        if self.model_type == 'MLP':
            # For MLP
            hidden_layers = params['hidden_layers']
            hidden_sizes = [params['hidden_size']] * hidden_layers
            return MLP(hidden_layers, hidden_sizes, params['activation_fn'])
        elif self.model_type == 'CNN':
            # For CNN
            num_conv_layers = params['num_conv_layers']
            filters = [params['filters']] * num_conv_layers
            hidden_layers = params['hidden_layers']
            hidden_sizes = [params['hidden_size']] * hidden_layers
            return CNN(num_conv_layers, filters, params['kernel_size'], params['pool_size'],
                       hidden_layers, hidden_sizes, params['activation_fn'])
    
    def create_optimizer(self, model, params):
        if params['optimizer'] == 'adam':
            return optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'rprop':
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
            
            # Use the number of epochs from the sampled parameters
            num_epochs = params['num_epochs']
            
            # Train model with early stopping
            model, history = train_model(model, train_loader, val_loader, optimizer, criterion, 
                                         num_epochs=num_epochs, patience=5, device=self.device)
            
            # Evaluate on validation set
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device=self.device)
            
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

