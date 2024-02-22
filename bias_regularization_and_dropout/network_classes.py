import numpy as np

class Regularization:
    def __init__(self, l2_lambda=0.01):
        self.l2_lambda = l2_lambda

    def l2_regularization_loss(self, weights):
        return 0.5 * self.l2_lambda * np.sum(weights ** 2)

    def l2_regularization_gradient(self, weights):
        return self.l2_lambda * weights

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_propagation(self, inputs):
        if self.dropout_rate > 0:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask
        else:
            return inputs

    def backward_propagation(self, gradient):
        if self.dropout_rate > 0:
            return gradient * self.mask
        else:
            return gradient

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
    def __init__(self, input_dimensions, output_dimensions, activation_type, regularization=None, dropout=None):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.activation_type = activation_type
        self.activation = Activation_Functions(activation_type)
        self.neurons = [Neuron(input_dimensions) for _ in range(output_dimensions)]
        self.bias = np.random.randn()
        self.regularization = regularization
        self.dropout = dropout

    def forward_propagation(self, inputs, targets, training=True):
        outputs = []
        for neuron in self.neurons:
            neuron.weights = np.random.rand(self.input_dimensions)
            neuron.bias = self.bias
            neuron.layer = self
            if training:
                inputs = self.dropout.forward_propagation(inputs)
            neuron.neuron(inputs)
            outputs.append(neuron.output)
        outputs = np.array(outputs)
        loss = np.mean(np.square(outputs - targets))
        if self.regularization:
            for neuron in self.neurons:
                loss += self.regularization.l2_regularization_loss(neuron.weights)
        return outputs, loss

class MultiLayerPerceptron:
    def __init__(self, input_dimensions, hidden_layer_sizes, output_size, activation_types, regularization=None, dropout=None):
        self.input_dimensions = input_dimensions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation_types = activation_types
        self.regularization = regularization
        self.dropout = dropout
        self.layers = []

        # Create input layer
        input_layer = Layer(input_dimensions, hidden_layer_sizes[0], activation_types[0], regularization, dropout)
        self.layers.append(input_layer)

        # Create hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            hidden_layer = Layer(hidden_layer_sizes[i], hidden_layer_sizes[i + 1], activation_types[i + 1], regularization, dropout)
            self.layers.append(hidden_layer)

        # Create output layer
        output_layer = Layer(hidden_layer_sizes[-1], output_size, activation_types[-1], regularization, dropout)
        self.layers.append(output_layer)

    def forward_propagation(self, inputs, targets, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs, loss = layer.forward_propagation(outputs, targets, training)
        return outputs, loss
