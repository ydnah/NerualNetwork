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
        # Storing inputs for back prop
        self.inputs = inputs
        # inputs.weights + bias
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        # Derivative with respect to weights = inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        # 
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
         # Derivative with respect to inputs = weights
        self.dinputs = np.dot(dvalues, self.weights.T)


# Creating a class for the ReLU activation function
class ReLU_Activation:
    # Forward pass
    def forward(self, inputs):
        # Store inputs for back prop
        self.inputs = inputs
        # Returns 0 if the input is less than or equal to 0 or the input if it is greater then 0
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        # Store inputs as modifiying original variable
        self.dinputs = dvalues.copy()
        # 
        self.dinputs[self.inputs <= 0] = 0
        
# Creating a class for the Softmax activation function
class Softmax:
    # Forward pass
    def forward(self, inputs):
        # Subtract the largest number from the input as subtract from each value
        # This is to protect against 'exploding values'
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Return normalised probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_ouput, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_ouput = single_ouput.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_ouput) - np.dot(single_ouput, single_ouput.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = -y_true / dvalues
        # Normalize the gradient to make the sum's magnitude invarient to the number of samples
        self.dinputs = self.dinputs / samples
    
class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Categorical_Cross_Entropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SDG:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

        
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam():
    def __init__(self, learning_rate=0.001, decay=0., epsiolon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsiolon = epsiolon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsiolon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsiolon)
        
        
    
    def post_update_params(self):
        self.iterations += 1
        
    
        
X, y = spiral_data(samples=100, classes = 3)

dense1 = Dense_Layer(2, 64)
activation1 = ReLU_Activation()
dense2 = Dense_Layer(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

# Train in loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}' +
        f'lr : {optimizer.current_learning_rate}')
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
