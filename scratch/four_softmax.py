# softmax is like smilified: input -> exponentiate -> normalize ->  output.

import numpy as np 

layer_outputs= [[4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_outputs)
exp_sum = np.sum(exp_values, axis=1, keepdims=True) # axis = 0 -> col wise, axis = 1 row wise, keepdims = 1/ture retain dim
norm_values = exp_values / exp_sum
print(norm_values)
