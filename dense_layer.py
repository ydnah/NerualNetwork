from typing import Any
import numpy as np
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

# Creating a class for the Dense layer
class Dense_Layer:
    # Initialize values
    def __init__(self, n_inputs, n_neurons):
        # creating a weights array of random values of shape number of inputs by number of neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Creating a bias row vector of 0's, of shape 1 by number of neurons
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass
    def forward(self, inputs):
        # inputs.weights + bias
        self.output = np.dot(inputs, self.weights) + self.biases


# Creating a class for the ReLU activation function
class ReLU_Activation:
    # Forward pass
    def forward(self, inputs):
        # Returns 0 if the input is less than or equal to 0 or the input if it is greater then 0
        self.output = np.maximum(0, inputs)
        
# Creating a class for the Softmax activation function
class Softmax:
    # Forward pass
    def forward(self, inputs):
        # Subtract the largest number from the input as subtract from each value
        # This is to protect against 'exploding values'
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Return normalised probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Creating a common loss class
# Many loss functions contain the same common operations
class Loss:
    def calculate(self, output, y):
        # Calculate sample loss
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        # Return data loss
        return data_loss

# Creating class for Categorical Cross Entropy loss, inheritting from the common loss class
class Categorical_Cross_Entropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        # Catgeorical labels 
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            
        # One-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Losses
        confidence = -np.log(correct_confidences)
        return confidence
        
X, y = spiral_data(samples=100, classes = 3)

dense1 = Dense_Layer(2,3)
acivation1 = ReLU_Activation()
dense2 = Dense_Layer(3,3)
activation2 = Softmax()

dense1.forward(X)
acivation1.forward(dense1.output)
dense2.forward(acivation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])