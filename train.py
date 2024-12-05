import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from model import SimpleCNN, count_parameters, save_model

def save_transformed_images_grid(data_loader, save_path="transformed_images_grid.png"):
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Create a grid of images
    grid = make_grid(images[:16], nrow=4, normalize=True, pad_value=1)  # 4x4 grid
    
    # Convert the grid to a NumPy array
    np_grid = grid.permute(1, 2, 0).numpy()  # Rearrange dimensions for matplotlib (H, W, C)
    
    # Save the grid as an image
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
        transforms.ColorJitter(brightness=0.20, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.RandomRotation((-10, 10), fill=(0,)),
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

       # Show some transformed images
    print("Transformed Training Images:")
    save_transformed_images_grid(train_loader)
    
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
