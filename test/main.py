import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sample data
# Input: (1 x 9) vector - represents one flattened image
X = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Weights: (4 x 9) matrix - each row represents weights for one neuron
# Note: Your example had 9 columns per row (transposed from earlier example)
W = np.array([
    [0.1, -0.2,  0.3,  0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2,  0.3, -0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0,  0.1,  0.2,  0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, -0.1,  0.1,  0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
])

# Biases: (1 x 4) vector - one bias per neuron
b = np.array([0.1, 0.1, 0.1, 0.1])

def forward_layer(X, W, b):
    """
    Compute forward pass through one layer
    Args:
        X: Input (1 x 9)
        W: Weights (4 x 9)
        b: Biases (1 x 4)
    Returns:
        Output after sigmoid (1 x 4)
    """
    # Need to transpose W for correct matrix multiplication
    # (1 x 9) @ (9 x 4) = (1 x 4)
    z = np.dot(X, W.T) + b
    return sigmoid(z)

# Run the layer and print dimensions at each step
print("Dimensions:")
print(f"Input X: {X.shape}")
print(f"Weights W: {W.shape}")
print(f"Biases b: {b.shape}")

output = forward_layer(X, W, b)
print(f"Output: {output.shape}")

print("\nActual values:")
print(f"Output: {output}")

# Let's also compute one neuron manually to verify
first_neuron = sigmoid(np.dot(X, W[0]) + b[0])
print(f"\nFirst neuron computed manually: {first_neuron}")