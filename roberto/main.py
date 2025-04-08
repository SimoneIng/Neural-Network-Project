import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Impostazione del seed per la riproducibilità
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Controllo della disponibilità di GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo: {device}")

# Parametri del dataset
TRAINING_SIZE = 10000
TEST_SIZE = 2500
BATCH_SIZE = 64

# Caricamento e preparazione del dataset MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Media e deviazione standard di MNIST
])

# Caricamento del dataset completo
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Selezione di un sottoinsieme casuale per il training set e validation set
train_size = TRAINING_SIZE
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Selezione di un sottoinsieme casuale per il test set
test_indices = torch.randperm(len(test_dataset))[:TEST_SIZE]
test_subset = torch.utils.data.Subset(test_dataset, test_indices)

# Creazione dei dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_subset)}")

# Definizione della rete neurale Fully Connected (MLP)
class MLP(nn.Module):
    def __init__(self, hidden_layers, hidden_sizes, dropout_rate, activation_fn):
        super(MLP, self).__init__()
        
        # Input layer (784 = 28x28 pixels)
        layers = [nn.Flatten()]
        input_size = 28 * 28
        
        # Aggiunta dei layer nascosti
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_sizes[i]))
            
            # Funzione di attivazione
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
        
        # Output layer (10 classi)
        layers.append(nn.Linear(input_size, 10))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Definizione della rete neurale Convoluzionale (CNN)
class CNN(nn.Module):
    def __init__(self, num_conv_layers, filters, kernel_size, pool_size, hidden_layers, 
                 hidden_sizes, dropout_rate, activation_fn):
        super(CNN, self).__init__()
        
        conv_layers = []
        in_channels = 1  # MNIST ha un solo canale (scala di grigi)
        
        # Creazione dei layer convoluzionali
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, filters[i], kernel_size))
            
            # Funzione di attivazione
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
        
        # Calcolo della dimensione dopo i layer convoluzionali
        # Per MNIST (28x28) con kernel_size=3, pool_size=2
        size_after_conv = 28
        for i in range(num_conv_layers):
            # Convoluzione: output = (input - kernel_size + 1)
            size_after_conv = size_after_conv - kernel_size + 1
            # Max pooling: output = input / pool_size
            size_after_conv = size_after_conv // pool_size
        
        # Dimensione input del primo layer fully connected
        self.flat_size = filters[-1] * (size_after_conv ** 2)
        
        # Layer fully connected
        fc_layers = [nn.Flatten()]
        input_size = self.flat_size
        
        for i in range(hidden_layers):
            fc_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            
            # Funzione di attivazione
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

# Funzione di addestramento
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, num_epochs=10):
    model = model.to(device)
    best_val_acc = 0.0
    best_model_weights = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
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
        
        # Valutazione sul validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        # Salvataggio del miglior modello
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
        
        # Aggiornamento della history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Caricamento dei pesi del miglior modello
    model.load_state_dict(best_model_weights)
    return model, history

# Funzione di valutazione
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

# Funzione per testare il modello e ottenere il report di classificazione
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
    
    # Matrice di confusione
    cm = confusion_matrix(all_labels, all_preds)
    
    # Report di classificazione
    report = classification_report(all_labels, all_preds, digits=4)
    
    # Accuratezza
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return accuracy, cm, report

# Plot della matrice di confusione
def plot_confusion_matrix(cm, title='Matrice di Confusione'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predetto')
    plt.ylabel('Reale')
    plt.show()

# Plot della history di addestramento
def plot_training_history(history, title='Training History'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Implementazione della Random Search
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
        return params
    
    def create_model(self, params):
        if self.model_type == 'MLP':
            # Per MLP
            hidden_layers = params['hidden_layers']
            hidden_sizes = [params['hidden_size']] * hidden_layers
            return MLP(hidden_layers, hidden_sizes, params['dropout_rate'], params['activation_fn'])
        elif self.model_type == 'CNN':
            # Per CNN
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
        elif params['optimizer'] == 'sgd':
            return optim.SGD(model.parameters(), lr=params['learning_rate'], 
                            momentum=params.get('momentum', 0), 
                            weight_decay=params.get('weight_decay', 0))
    
    def search(self, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        best_params = None
        best_model = None
        
        for i in range(self.num_trials):
            params = self.sample_params()
            print(f"\nTrial {i+1}/{self.num_trials}")
            print(f"Parameters: {params}")
            
            # Creazione del modello e dell'ottimizzatore
            model = self.create_model(params)
            optimizer = self.create_optimizer(model, params)
            
            # Addestramento del modello
            model, history = train_model(model, train_loader, val_loader, optimizer, criterion, 
                                         num_epochs=self.num_epochs)
            
            # Valutazione sul validation set
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            
            # Registrazione dei risultati
            self.results.append({
                'params': params,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            # Aggiornamento del miglior modello
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params
                best_model = model
        
        return best_model, best_params, self.results

# Spazio di ricerca per MLP
mlp_param_space = {
    'hidden_layers': (1, 4),           # Numero di layer nascosti
    'hidden_size': [64, 128, 256, 512], # Dimensione dei layer nascosti
    'dropout_rate': (0.0, 0.5),        # Tasso di dropout
    'learning_rate': (0.0001, 0.01),   # Learning rate
    'optimizer': ['adam'],             # Ottimizzatore (scegliamo ADAM come da requisiti)
    'activation_fn': ['relu', 'tanh', 'leaky_relu'], # Funzione di attivazione
    'weight_decay': (0.0, 0.001)       # Regolarizzazione L2
}

# Spazio di ricerca per CNN
cnn_param_space = {
    'num_conv_layers': (1, 3),          # Numero di layer convoluzionali
    'filters': [16, 32, 64, 128],       # Numero di filtri
    'kernel_size': [3, 5],              # Dimensione del kernel
    'pool_size': [2],                   # Dimensione del pool
    'hidden_layers': (0, 2),            # Numero di layer fully connected
    'hidden_size': [64, 128, 256, 512], # Dimensione dei layer fully connected
    'dropout_rate': (0.0, 0.5),         # Tasso di dropout
    'learning_rate': (0.0001, 0.01),    # Learning rate
    'optimizer': ['adam'],              # Ottimizzatore (scegliamo ADAM come da requisiti)
    'activation_fn': ['relu', 'tanh', 'leaky_relu'], # Funzione di attivazione
    'weight_decay': (0.0, 0.001)        # Regolarizzazione L2
}

# Numero di prove per la random search
NUM_TRIALS = 10
# Numero di epoche per l'addestramento di ogni modello
NUM_EPOCHS = 10

# Esecuzione della random search per MLP
print("\n===== Random Search per MLP =====")
mlp_random_search = RandomSearch('MLP', mlp_param_space, num_trials=NUM_TRIALS, num_epochs=NUM_EPOCHS)
best_mlp, best_mlp_params, mlp_results = mlp_random_search.search(train_loader, val_loader)

# Esecuzione della random search per CNN
print("\n===== Random Search per CNN =====")
cnn_random_search = RandomSearch('CNN', cnn_param_space, num_trials=NUM_TRIALS, num_epochs=NUM_EPOCHS)
best_cnn, best_cnn_params, cnn_results = cnn_random_search.search(train_loader, val_loader)

# Test del miglior modello MLP
print("\n===== Valutazione del miglior MLP sul Test Set =====")
mlp_accuracy, mlp_cm, mlp_report = test_model(best_mlp, test_loader)
print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
print("MLP Classification Report:")
print(mlp_report)
plot_confusion_matrix(mlp_cm, title='Matrice di Confusione - MLP')

# Test del miglior modello CNN
print("\n===== Valutazione del miglior CNN sul Test Set =====")
cnn_accuracy, cnn_cm, cnn_report = test_model(best_cnn, test_loader)
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")
print("CNN Classification Report:")
print(cnn_report)
plot_confusion_matrix(cnn_cm, title='Matrice di Confusione - CNN')

# Confronto dei risultati
print("\n===== Confronto dei risultati =====")
print(f"MLP Best Parameters: {best_mlp_params}")
print(f"CNN Best Parameters: {best_cnn_params}")
print(f"MLP Test Accuracy: {mlp_accuracy:.4f}")
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

# Plot dei risultati della random search
def plot_random_search_results(results, title):
    # Ordina i risultati per accuratezza di validazione
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    acc_values = [r['val_acc'] for r in sorted_results]
    trial_indices = range(1, len(sorted_results) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.bar(trial_indices, acc_values)
    plt.title(title)
    plt.xlabel('Trial (ordinati per accuratezza)')
    plt.ylabel('Validation Accuracy')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_random_search_results(mlp_results, 'MLP Random Search Results')
plot_random_search_results(cnn_results, 'CNN Random Search Results')

# Confronto finale
labels = ['MLP', 'CNN']
test_accuracies = [mlp_accuracy, cnn_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(labels, test_accuracies)
plt.title('Confronto delle Accuratezze sul Test Set')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Aggiungi i valori sopra le barre
for i, v in enumerate(test_accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.show()