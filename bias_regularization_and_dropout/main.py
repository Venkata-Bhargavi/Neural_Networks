import numpy as np

from network_classes import Regularization,MultiLayerPerceptron


def main():
    # Example usage with L2 regularization
    input_dimensions = 2
    hidden_layer_sizes = [3, 4]  # Two hidden layers with 3 and 4 neurons respectively
    output_size = 1
    activation_types = ["relu", "sigmoid"]  # Activation functions for each layer
    reg_lambda = 0.01  # Regularization strength
    # choose "l1" or "l2" regularization
    regularization = Regularization(reg_type="l2", reg_lambda=reg_lambda)

    mlp = MultiLayerPerceptron(input_dimensions, hidden_layer_sizes, output_size, activation_types, regularization)

    # Initialize input data and targets
    num_samples = 10
    inputs = np.random.randint(2, size=(num_samples, input_dimensions))
    targets = np.random.randint(2, size=(num_samples, output_size))

    # Perform forward propagation
    outputs, loss = mlp.forward_propagation(inputs, targets)
    print("inputs: ", inputs)
    print("Outputs:", outputs)
    print("target: ", targets)
    print("Loss:", loss)


if __name__ == "__main__":
    main()


import numpy as np
from network_classes import MultiLayerPerceptron, Regularization, Dropout

def main():
    # Example usage:
    input_dimensions = 2
    hidden_layer_sizes = [3, 4]  # Two hidden layers with 3 and 4 neurons respectively
    output_size = 1
    activation_types = ["relu", "sigmoid"]  # Activation functions for each layer

    # Initialize regularization and dropout instances
    l2_regularization = Regularization(l2_lambda=0.01)
    dropout = Dropout(dropout_rate=0.2)

    # Create MultiLayerPerceptron with regularization and dropout
    mlp = MultiLayerPerceptron(input_dimensions, hidden_layer_sizes, output_size, activation_types, l2_regularization, dropout)

    # Initialize input data_open_source and targets
    num_samples = 10
    inputs = np.random.randint(2, size=(num_samples, input_dimensions))
    targets = np.random.randint(2, size=(num_samples, output_size))

    # Perform forward propagation with regularization and dropout
    outputs, loss = mlp.forward_propagation(inputs, targets)
    print("inputs: ", inputs)
    print("Outputs:", outputs)
    print("target: ",targets)
    print("Loss:", loss)

if __name__ == "__main__":
    main()
