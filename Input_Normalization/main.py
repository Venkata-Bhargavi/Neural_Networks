import numpy as np

from network_classes import Regularization,Dropout,MultiLayerPerceptron

def main():
    # Example usage:
    input_dimensions = 2
    hidden_layer_sizes = [3, 4]  # Two hidden layers with 3 and 4 neurons respectively
    output_size = 1
    activation_types = ["relu", "sigmoid"]  # Activation functions for each layer

    # Choose normalization type
    normalization_type = "min_max"  # Choose either "z_score" or "min_max"

    # Initialize regularization and dropout instances
    regularization = Regularization(l2_lambda=0.01)
    dropout = Dropout(dropout_rate=0.2)

    # Create MultiLayerPerceptron with normalization, regularization, and dropout
    mlp = MultiLayerPerceptron(input_dimensions, hidden_layer_sizes, output_size, activation_types, normalization_type, regularization, dropout)

    # Initialize input data and targets
    num_samples = 10
    inputs = np.random.randint(2, size=(num_samples, input_dimensions))
    targets = np.random.randint(2, size=(num_samples, output_size))

    # Perform forward propagation with normalization, regularization, and dropout
    outputs, loss = mlp.forward_propagation(inputs, targets)
    print("inputs: ", inputs)
    print("Outputs:", outputs)
    print("target: ", targets)
    print("Loss:", loss)

if __name__ == "__main__":
    main()