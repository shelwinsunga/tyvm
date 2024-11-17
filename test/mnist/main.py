import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np

# Define the neural network
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),  # Input layer (28x28 = 784) to first hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),   # First hidden layer to second hidden layer
            nn.ReLU(),
            nn.Linear(64, 10)     # Second hidden layer to output layer
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        # Simple normalization (L2 norm)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=False
)

# Initialize the network
model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# After training, export the weights
def export_weights(model):
    state_dict = model.state_dict()
    weights_dict = {
        'layer1': {
            'weights': state_dict['layers.0.weight'].cpu().numpy().tolist(),
            'biases': state_dict['layers.0.bias'].cpu().numpy().tolist()
        },
        'layer2': {
            'weights': state_dict['layers.2.weight'].cpu().numpy().tolist(),
            'biases': state_dict['layers.2.bias'].cpu().numpy().tolist()
        },
        'layer3': {
            'weights': state_dict['layers.4.weight'].cpu().numpy().tolist(),
            'biases': state_dict['layers.4.bias'].cpu().numpy().tolist()
        }
    }
    
    with open('mnist_weights.json', 'w') as f:
        json.dump(weights_dict, f)

# Add this after the training loop and testing
export_weights(model)
print("Weights exported to mnist_weights.json")