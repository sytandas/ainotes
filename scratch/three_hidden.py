import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# dataset
def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y 
# layer
class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU 
class activation_relu:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

class activation_solftmax:
    def forward(self, inputs):
        exp_value = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # input - max(input) -> for avoiding overflow
        norm_value = np.sum(exp_value, axis=1, keepdims=True)
        self.outuput = exp_value / norm_value

X, y = spiral_data(samples=100, classes=3)

dense1 =Layer_dense(2,3)
activation1 = activation_relu()

dense2 = Layer_dense(3, 3)
activation2 = activation_solftmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.outputs)
activation2.forward(dense2.output)

print(activation2.outuput[:5])