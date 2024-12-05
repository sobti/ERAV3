import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from model import SimpleCNN, count_parameters, save_model

def save_transformed_images_grid(data_loader, save_path="outputs/transformed_images_grid.png"):
       # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    grid = make_grid(images[:16], nrow=4, normalize=True, pad_value=1)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid, cmap="gray")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()
    print(f"Transformed image grid saved to {save_path}")


def train_model():

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
      # Train Phase transformations
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0, hue=0.5),
        transforms.RandomRotation((-10, 10), fill=(0,)),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std are tuples
    ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    assert len(train_dataset) == 60000, "Training dataset should have 60,000 samples"
    assert len(test_dataset) == 10000, "Test dataset should have 10,000 samples"


       # Show some transformed images
    print("Transformed Training Images:")
    save_transformed_images_grid(train_loader, save_path="outputs/transformed_images_grid.png")

    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        assert output.shape == (64, 10), f"Expected output shape (64, 10), but got {output.shape}"
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Train Accuracy: {accuracy:.2f}%')
    
    # Save model with timestamp
    model_filename = save_model(model, accuracy)
    model_filename=''
    
    return model, accuracy, model_filename

if __name__ == '__main__':
    train_model()
