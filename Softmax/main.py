from Activation_classes import NeuralNetwork
import numpy as np

def main():
    # Example usage:
    input_size = 3  # Number of features
    hidden_size = 5  # Number of hidden units
    output_size = 3  # Number of classes

    # Create a neural network
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Example input data representing features for 2 instances
    input_data = np.array([
        [1.2, 0.5, 0.8],  # Features for instance 1
        [0.9, 1.0, 1.5]   # Features for instance 2
    ])

    # Simulate the neural network
    raw_scores, predictions = model.forward(input_data)

    # Output the predictions
    print("Input Data:\n", input_data)
    print("\nRaw Scores:\n", raw_scores)
    print("\nPredicted Probabilities:\n", predictions)

if __name__ == "__main__":
    main()