import torch
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
from PIL import Image

# Load a single MNIST image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    transform=transform
)

# Get first image and save both raw image and normalized array
image, label = test_dataset[0]
image_flat = image.flatten().numpy()

# Save the flattened normalized array
with open('mnist_sample.json', 'w') as f:
    json.dump(image_flat.tolist(), f)

# Save the original image (before normalization)
original_image = test_dataset.data[0].numpy()
Image.fromarray(original_image).save('mnist_sample.png')

print(f"Sample image (digit {label}) exported to mnist_sample.json and mnist_sample.png")

# Manual forward pass
def relu(x):
    return np.maximum(0, x)

def normalize(x):
    return x / np.linalg.norm(x)

# Load weights
with open('mnist_weights.json', 'r') as f:
    weights = json.load(f)

# Forward pass
x = image_flat
# Layer 1
x = np.dot(weights['layer1']['weights'], x) + weights['layer1']['biases']
x = relu(x)
# Layer 2
x = np.dot(weights['layer2']['weights'], x) + weights['layer2']['biases']
x = relu(x)
# Layer 3
x = np.dot(weights['layer3']['weights'], x) + weights['layer3']['biases']
x = normalize(x)

print(f"Predicted probabilities: {x}")
print(f"Predicted digit: {np.argmax(x)}")
print(f"Actual digit: {label}")