import numpy as np

class Neuron:
    def __init__(self, input_dimensions):
        self.weights = None
        self.bias = None
        self.input_dimensions = input_dimensions
        self.aggregate_signal = None
        self.activation = None
        self.output = None
        self.layer = None

    def neuron(self, inputs):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")
        if inputs.shape[-1] != self.input_dimensions:
            raise ValueError(f"Inputs shape ({inputs.shape}) is incompatible with neuron input dimensions ({self.input_dimensions})")
        self.aggregate_signal = np.sum(np.dot(inputs, self.weights) + self.bias)
        self.activation = self.layer.activation(self.aggregate_signal)
        self.output = self.activation

class Activation_Functions:
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        if self.type == "relu":
            return np.maximum(0, inputs)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-inputs))
        elif self.type == "tanh":
            return np.tanh(inputs)
        elif self.type == "softmax":
            exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Invalid activation function: {self.type}, Choose a valid activation function")

class Layer:
    def __init__(self, input_dimensions, output_dimensions, activation_type):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.activation_type = activation_type
        self.activation = Activation_Functions(activation_type)
        self.neurons = [Neuron(input_dimensions) for _ in range(output_dimensions)]
        self.bias = np.random.randn()

    def forward_propagation(self, inputs, targets):
        outputs = []
        for neuron in self.neurons:
            neuron.weights = np.random.rand(self.input_dimensions)
            neuron.bias = self.bias
            neuron.layer = self
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        outputs = np.array(outputs)
        loss = np.mean(np.square(outputs - targets))
        return outputs, loss

    def backward_propagation(self, gradients, learning_rate):
        grad_weights = np.zeros_like(self.neurons[0].weights)
        grad_bias = 0

        for i, neuron in enumerate(self.neurons):
            grad_activation = gradients

            grad_aggregate_signal = grad_activation * self.activation_derivative(neuron.aggregate_signal)

            grad_weights += np.outer(grad_aggregate_signal, neuron.output)
            grad_bias += np.sum(grad_aggregate_signal)

            gradients = np.dot(neuron.weights, grad_aggregate_signal)

        self.bias -= learning_rate * grad_bias
        for neuron in self.neurons:
            neuron.weights -= learning_rate * grad_weights

        return gradients

    def activation_derivative(self, x):
        if self.activation_type == "relu":
            return np.where(x > 0, 1, 0)
        elif self.activation_type == "sigmoid":
            return self.activation(x) * (1 - self.activation(x))
        elif self.activation_type == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_type == "softmax":
            raise NotImplementedError("Derivative of softmax function not implemented")
        else:
            raise ValueError(f"Invalid activation function: {self.activation_type}")

class MultiLayerPerceptron:
    def __init__(self, input_dimensions, hidden_layer_sizes, output_size, activation_types):
        self.input_dimensions = input_dimensions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation_types = activation_types
        self.layers = []

        # Create input layer
        input_layer = Layer(input_dimensions, hidden_layer_sizes[0], activation_types[0])
        self.layers.append(input_layer)

        # Create hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            hidden_layer = Layer(hidden_layer_sizes[i], hidden_layer_sizes[i + 1], activation_types[i + 1])
            self.layers.append(hidden_layer)

        # Create output layer
        output_layer = Layer(hidden_layer_sizes[-1], output_size, activation_types[-1])
        self.layers.append(output_layer)

    def forward_propagation(self, inputs, targets):
        outputs = inputs
        for layer in self.layers:
            outputs, loss = layer.forward_propagation(outputs, targets)
        return outputs, loss

    def backward_propagation(self, inputs, targets, learning_rate):
        gradients = 1
        for layer in reversed(self.layers):
            gradients = layer.backward_propagation(gradients, learning_rate)

input_dimensions = 2
hidden_layer_sizes = [3, 4]  # Two hidden layers with 3 and 4 neurons respectively
output_size = 1
activation_types = ["relu", "sigmoid"]  # Activation functions for each layer

mlp = MultiLayerPerceptron(input_dimensions, hidden_layer_sizes, output_size, activation_types)

# Initialize input data and targets
num_samples = 10
inputs = np.random.randint(2, size=(num_samples, input_dimensions))
targets = np.random.randint(2, size=(num_samples, output_size))

# Perform forward propagation
outputs, loss = mlp.forward_propagation(inputs, targets)
print("inputs: ", inputs)
print("Outputs:", outputs)
print("target: ",targets)
print("Loss:", loss)

# Perform backpropagation
learning_rate = 0.01
mlp.backward_propagation(inputs, targets, learning_rate)

# Perform forward propagation again to check if loss has decreased
outputs, loss = mlp.forward_propagation(inputs, targets)
print("Updated Loss:", loss)
