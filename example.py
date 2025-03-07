import torch.optim.sgd
from dataset import train_loader, validation_loader, test_loader, train_size, validation_size, test_size, batch_size
import torch 
from torch import nn 

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 0.01

class DeepNeuralNetwork(nn.Module): 
    def __init__(self): 
        #chiamata al costruttore 
        super(DeepNeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        self.first_hidden_layer = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(), 
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 256),
            nn.ReLU(), 
            nn.Linear(256, 10)
        )

    def forward(self, x): 
        x = self.flatten(x)
        return self.first_hidden_layer(x)
    
model = DeepNeuralNetwork().to(device) 

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer): 
    
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
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    
    return correct



def train(epochs): 

    best_accurancy = 0
    best_accurancy_epoch = -1

    for t in range(epochs): 
        print(f"Epoch: {t+1}\n")

        train_loop(train_loader, model, loss_fn, optimizer)
        acc = test_loop(validation_loader, model, loss_fn)
        
        if(acc > best_accurancy): 
            best_accurancy = acc
            best_accurancy_epoch = t+1 
        
    print("Done...\n")
    print(f"Best Accurancy: {(100*best_accurancy):>0.1f}%, Epoch: {best_accurancy_epoch} \n")



# Actual Training
train(15)


    