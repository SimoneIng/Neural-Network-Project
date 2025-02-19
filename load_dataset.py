import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plot 
import numpy as np

transform = transforms.ToTensor() 

tr_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
ts_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = 8000 
validation_size = 2000

train_dataset, validation_dataset = random_split(Subset(tr_ds, range(10000)), [train_size, validation_size])
test_dataset = Subset(ts_ds, range(2500))

print("Dataset Caricato... \n")

# DataLoader per iterare sui dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("DataLoader pronti...\n")

image, label = train_dataset[0]


print(f"shape: {label.shape}")