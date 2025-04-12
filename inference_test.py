# In un altro script o sessione, carica il modello
import time
import torch
from torchvision import datasets, transforms
from src.utils.inference import inference, load_model

transform = transforms.Compose([
    # transforms.Resize((28, 28)),  # Ridimensiona a 28x28
    transforms.ToTensor(),  # Converti in tensore
    transforms.Normalize((0.1307,), (0.3081,))  # Applica la stessa normalizzazione di MNIST
])


def get_mnist_digit(digit, num_samples=1, return_first=True):
    """
    Ottiene uno o più esempi di un dato numero dal dataset MNIST

    Args:
        digit: il numero da trovare (0-9)
        num_samples: quanti esempi recuperare
        return_first: se True, restituisce solo il primo esempio trovato

    Returns:
        Un tensore di forma [num_samples, 1, 28, 28] se return_first=False
        Un tensore di forma [1, 1, 28, 28] se return_first=True
    """
    # Carica il dataset MNIST
    mnist_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Trova gli indici di tutte le immagini che rappresentano il numero richiesto
    digit_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == digit]

    if len(digit_indices) == 0:
        raise ValueError(f"Nessun esempio trovato per il numero {digit}")

    # Prendi i campioni richiesti (o solo il primo)
    if return_first:
        image, _ = mnist_dataset[digit_indices[0]]
        return image.unsqueeze(0)  # Aggiungi dimensione batch [1, 1, 28, 28]
    else:
        # Recupera gli esempi richiesti
        samples = []
        for i in range(min(num_samples, len(digit_indices))):
            image, _ = mnist_dataset[digit_indices[i]]
            samples.append(image.unsqueeze(0))
        return torch.cat(samples, dim=0)  # [num_samples, 1, 28, 28]


def print_mnist_digit(image_tensor: torch.Tensor):
    """
    Stampa una rappresentazione testuale dell'immagine MNIST

    Args:
        image_tensor: tensore PyTorch di forma [1, 1, 28, 28] o [1, 28, 28]
    """
    # Converti il tensore in numpy array e rimuovi le dimensioni non necessarie
    if image_tensor.dim() == 4:
        image = image_tensor[0, 0].cpu().numpy()
    elif image_tensor.dim() == 3:
        image = image_tensor[0].cpu().numpy()
    else:
        image = image_tensor.cpu().numpy()

    # Scala i valori per una migliore visualizzazione testuale
    # MNIST ha valori normalizzati tra 0 e 1
    image = (image * 9).astype(int)

    # Stampa l'immagine come caratteri ASCII, dove valori più alti sono più scuri
    chars = " .:-=+*#%@"  # Caratteri ASCII dal più chiaro al più scuro
    for row in image:
        print("".join(chars[min(9, max(0, val))] for val in row))


device = torch.device("cuda")

print(f"Device utilizzato: {device}")

start_time = time.time()
loaded_model = load_model("./src/experiments/results/cnn_model.pt", device)
print(f"Caricamento modello: {time.time() - start_time:.4f} secondi")

# test_image = get_mnist_digit(8, return_first=True) 
# prediction = inference(loaded_model, test_image.to(device))

# print(prediction.shape)
# print(prediction.item())

for i in range(10):
    digit = i
    num_samples = 10000
    correct = 0

    print(f"DIGIT: {digit}")
    start_time = time.time()
    test_images = get_mnist_digit(digit, num_samples=num_samples, return_first=False)  # Esempio di un'immagine MNIST
    print(f"Caricamento data set: {time.time() - start_time:.4f} secondi")

    start_time = time.time()
    for i in range(test_images.shape[0]):
        # print_mnist_digit(test_images[i])
        # print("#################")
        prediction = inference(loaded_model, test_images[i].unsqueeze(0).to(device))
        if prediction.item() == digit:
            correct += 1
    print(f"Inferenza: {time.time() - start_time:.4f} secondi")

    print(f"{correct}/{test_images.shape[0]}")
    print("\n")
