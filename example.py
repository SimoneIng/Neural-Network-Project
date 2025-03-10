import torch.optim.sgd
from dataset import train_loader, validation_loader, test_loader, train_size, validation_size, test_size, batch_size
import torch 
from torch import nn 
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 0.001

class DeepNeuralNetwork(nn.Module): 
    def __init__(self): 
        #chiamata al costruttore 
        super(DeepNeuralNetwork, self).__init__()
        
        self.flatten = nn.Flatten()
        self.first_hidden_layer = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(), 
            nn.Linear(256, 256), 
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
    
    avg_loss = 0
    loss_sum = 0
    
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
        
        loss_sum += loss
        
    avg_loss = loss_sum / size
    return avg_loss

def validation_loop (dataloader, model, loss_fn):
    
    size = len(dataloader.dataset)
    model.eval()
    batch_nums = len(dataloader)
    val_loss, correct = 0, 0 
    
    # valutazione del modello 
    with torch.no_grad():
        for X, y in dataloader: 
            X, y = X.to(device), y.to(device)  # Move to GPU
            
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    val_loss /= batch_nums
    correct /= size 
    
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    
    return val_loss


def train(epochs): 
    
    train_losses = [] 
    validation_losses = []

    for t in range(epochs): 
        print(f"Epoch: {t+1}\n")

        avg_tloss = train_loop(train_loader, model, loss_fn, optimizer)
        train_losses.insert(t, avg_tloss)
        avg_vloss = validation_loop(validation_loader, model, loss_fn)
        validation_losses.insert(t, avg_vloss)
        
    print("Done...\n")
    
    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Error")
    plt.plot(range(1, epochs + 1), validation_losses, label="Validation Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Training Vs Validation Loss")
    plt.show()



def testing(dataloader, model, loss_fn):
    
    size = len(dataloader.dataset) 
    model.eval()
    batch_nums = len(dataloader)
    test_loss, correct = 0, 0 
    
    with torch.no_grad(): 
        for X, y in dataloader: 
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
               
    test_loss /= batch_nums
    correct /= size 
    
    print(f"Testing Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 





# Actual Training
train(100)
testing(test_loader, model, loss_fn)


    