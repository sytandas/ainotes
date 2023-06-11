import numpy as np

inputs = [1.2, 3.2, 8.8]

weights = [[0.8, 0.45, -0.3], 
           [0.6, 0.98, 0.2], 
           [0.82, 0.27, 0.11]]

biases = [0.03, 0.4, 2.0] 

output = np.dot(weights, inputs) + biases
print(output)