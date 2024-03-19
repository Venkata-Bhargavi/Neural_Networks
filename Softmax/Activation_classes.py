import numpy as np

class SoftmaxActivation:
    def __init__(self):
        pass

    def softmax(self, z):
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, X):
        return np.dot(X, self.weights) + self.biases

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = DenseLayer(input_size, hidden_size)
        self.layer2 = DenseLayer(hidden_size, output_size)

        self.activation = SoftmaxActivation()

    def forward(self, X):
        hidden_output = self.layer1.forward(X)
        hidden_activation = self.activation.softmax(hidden_output)
        output = self.layer2.forward(hidden_activation)
        probabilities = self.activation.softmax(output)  # Apply softmax to the output layer
        return output, probabilities
