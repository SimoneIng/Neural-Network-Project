import torch 
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt 
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

#! Defining Hyperparameters

param_space = {
    "Hidden Layers": [1, 2, 4], 
    "batch_size": [32, 64, 128],
    "number_of_neurons": [64, 128, 256], 
    "learning_rate": [0.01, 0.001, 0.0001], 
    "epochs": [5, 10, 15], 
    "activation_fn": [nn.Sigmoid, nn.Softmax, nn.ReLU] 
}


#! Defining Neural Networks

class SingleLayerNN(nn.Module):
    def __init__(self, hidden_nodes, activation_fn): 
        super(SingleLayerNN, self).__init__()
        
        self.flatten = nn.Flatten()
        self.hidden_layer = nn.Sequential(
            nn.Linear(28*28, hidden_nodes),
            activation_fn(),
            nn.Linear(hidden_nodes, 10)
        )
        
    def forward(self, x): 
        x = self.flatten(x)
        output = self.hidden_layer(x)
        return output        

         

class DoubleLayerNN(nn.Module):
    def __init__(self): 
        super(DoubleLayerNN, self).__init__()
        
        self.flatten = nn.Flatten()


class ThreeLayerNN(nn.Module):
    def __init__(self): 
        super(ThreeLayerNN, self).__init__()
        
        self.flatten = nn.Flatten()



#! Loading the Dataset

train_size = 8000 
validation_size = 2000
test_size = 2500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
]) 

tr_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ts_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


train_dataset, validation_dataset = random_split(Subset(tr_ds, range(10000)), [train_size, validation_size])
test_dataset = Subset(ts_ds, range(2500))

def train_loop(dataloader, model, loss_fn, optimizer, batch_size): 
    
    size = len(dataloader.dataset)
    
    model.train()
    
    for batch, (X, y) in enumerate(dataloader): 
        X, y = X.to(device), y.to(device)  # Move to GPU
        
        # Computazione Prediction e Loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * batch_size + len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop (dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    batch_nums = len(dataloader)
    test_loss, correct = 0, 0 
    
    # valutazione del modello 
    with torch.no_grad():
        for X, y in dataloader: 
            X, y = X.to(device), y.to(device)  # Move to GPU
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= batch_nums
    correct /= size 
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")






#! Random Search Training

def random_search(num_execution):
    best_model = None
    best_accurancy = 0 
    best_params = None
    
    for current_execution in range(num_execution): 

        # Selezione dei parametri casuali
        num_layers = random.choice(param_space["Hidden Layers"])
        neurons = random.choice(param_space['number_of_neurons'])
        batch_size = random.choice(param_space["batch_size"])
        epochs = random.choice(param_space['epochs'])
        learning_rate = random.choice(param_space['learning_rate'])
        activation_fn = random.choice(param_space['activation_fn'])
        
        
        # DataLoader per iterare sui dataset
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # creazione del modello 
        model = SingleLayerNN(neurons, activation_fn).to(device) 
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        print(f"Current Hyperparameters: \n neurons:{neurons}, batch Size:{batch_size}, epochs:{epochs}, learning Rate{learning_rate}, activation Function:{activation_fn} \n")
        
        # addestramento
        for t in range(epochs): 
            train_loop(train_loader, model, loss_fn, optimizer, batch_size)
            test_loop(validation_loader, model, loss_fn)
        print("Training Done for this setup...\n")   
    
    print("Random Search Done..\n")


random_search(3)