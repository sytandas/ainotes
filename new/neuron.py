import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input and output data
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])

# Define the hyperparameters
epochs = 5000
learning_rate = 0.1

# Define the weights
w1 = np.random.randn(3,3)
w2 = np.random.randn(3,1)

# Train the neural network
for i in range(epochs):
    # Forward propagation
    layer1 = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(layer1, w2))

    # Backpropagation
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    layer1_error = output_delta.dot(w2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update the weights
    w2 += layer1.T.dot(output_delta) * learning_rate
    w1 += X.T.dot(layer1_delta) * learning_rate

# Test the neural network
test_input = np.array([1,1,0])
layer1 = sigmoid(np.dot(test_input, w1))
output = sigmoid(np.dot(layer1, w2))

print(output)

