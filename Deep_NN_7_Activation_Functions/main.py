import numpy as np
from network_classes import MultiLayerPerceptron


def main():
    # Example usage:
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




if __name__ == "__main__":
    main()
