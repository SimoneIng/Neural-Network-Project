import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt 
import numpy as np

batch_size = 32

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

print("Dataset Caricato... \n")

# DataLoader per iterare sui dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("DataLoader pronti...\n")

