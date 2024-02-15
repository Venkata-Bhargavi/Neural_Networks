import numpy as np

# class Parameters:
#     def __init__(self, input_dimensions):
#         self.weights = np.random.rand(input_dimensions)
#         self.bias = np.random.randn()
#
#     def get_weights(self):
#         return self.weights
#
#     def get_bias(self):
#         return self.bias

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


