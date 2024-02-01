import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, activation):
        '''
        weights and bias initialization ------ This varies based on the input given in the main file
        :param activation: chosen activation function out of sigmoid, sigmoid derivative, relu, relu derivative
        '''
        # Initialize weights and biases using Xavier initialization
        np.random.seed(0)
        self.hidden_weights = np.random.randn(2, 2) * np.sqrt(2 / (2 + 2))  # Xavier initialization
        self.hidden_bias = np.random.randn(2)

        self.output_weights = np.random.randn(2, 1) * np.sqrt(2 / (2 + 1))  # Xavier initialization

        self.output_bias = np.random.randn()

        # Choose activation function for hidden layer
        if activation == 'sigmoid':
            self.activation_function = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        else:
            raise ValueError("Activation function not supported")

    def forward_propagation(self, inputs):
        '''

        :param inputs:
        :return: hidden_layer_output: This is the output of the hidden layer after applying the activation function
        to the linear combination of inputs, hidden weights, and biases. It represents the activations of the neurons in the hidden layer.
                output: This is the final output of the neural network after applying the activation function to
        the linear combination of hidden layer outputs, output weights, and output bias. It represents the predicted output of the neural network for the given inputs.
        '''
        # Forward propagation through the network
        hidden_layer_input = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = Activation.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, (self.output_weights)) + self.output_bias
        output = Activation.sigmoid(output_layer_input)

        return hidden_layer_output, output

    def backward_propagation(self, inputs, hidden_output, output, target, learning_rate):
        # Backpropagation
        output_error = target - output
        output_delta = output_error * Activation.sigmoid_derivative(output)

        # Transpose self.output_weights
        hidden_error = np.dot(output_delta, self.output_weights.T)

        hidden_delta = hidden_error * Activation.sigmoid_derivative(hidden_output)

        # Update output_weights
        self.output_weights += np.dot(hidden_output.T, output_delta) * learning_rate
        self.output_bias += np.sum(output_delta) * learning_rate

        # Update weights and biases
        self.hidden_weights += np.dot(inputs.T, hidden_delta) * learning_rate
        self.hidden_bias += np.sum(hidden_delta) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """
        function to train the network with input parameters
        :param X: input data
        :param y: target
        :param epochs: no of epochs to train
        :param learning_rate: learning rate hyperparameter
        :return: None but prints the loss after each 100 epochs
        """
        for epoch in range(epochs):
            # Forward propagation
            hidden_output, output = self.forward_propagation(X)

            # Backpropagation
            self.backward_propagation(X, hidden_output, output, y, learning_rate)

            # Calculate loss
            loss = np.mean((y - output) ** 2)
            if epoch%100 == 0: print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


