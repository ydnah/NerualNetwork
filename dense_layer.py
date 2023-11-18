import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes = 3)

dense1 = Dense_Layer(2,3)
dense1.forward(X)
