from helpers import Perceptron
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import ast  # Module to convert string representation of lists to actual lists

input_size = 20 * 20  # Each image has dimensions 20x20 pixels

# Create an instance of the Perceptron class with the calculated input size
perceptron = Perceptron(input_size=input_size)

# Initialize the parameters
perceptron.initialize_parameters()

# Check the initialized parameters
print("Initialized weights shape:", perceptron.weights.shape)
print("Initialized bias:", perceptron.bias)


# Define a custom converter function to safely convert string to list
def safe_string_to_list(string):
    '''

    :param string: string which has list as string
    :return: string
    '''
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        return string


# Define a custom converter function to convert string representation of lists to actual lists
def parse_pixel_matrix(pixel_matrix_str):
    # Remove '[' and ']' characters from the string and split by whitespaces
    pixel_values = pixel_matrix_str.strip('][').split()
    # Convert strings to integers and create a NumPy array
    pixel_array = np.array([int(value) for value in pixel_values])
    # Reshape the array to the original shape (20 x 20)
    # pixel_matrix = pixel_array.reshape((20, 20))
    return pixel_array

# Read the CSV file with the custom converter function
dataset = pd.read_csv("training_data.csv", converters={"pixel_matrix": parse_pixel_matrix})

# Extract pixel matrices and labels from the dataset
pixel_matrices = dataset['pixel_matrix'].values
labels = dataset['label'].values

# Convert pixel matrices to numpy array and normalize them
pixel_matrices = np.array([np.array(matrix) / 255.0 for matrix in pixel_matrices])

# Shuffle the data
indices = np.arange(len(labels))
np.random.shuffle(indices)
pixel_matrices = pixel_matrices[indices]
labels = labels[indices]


# Split data into training and test sets (80% training, 20% test)
train_inputs, test_inputs, train_labels, test_labels = train_test_split(pixel_matrices, labels, test_size=0.2, stratify=labels)

# Check the shapes of the training and test sets
print("Train inputs shape:", train_inputs.shape)
print("Test inputs shape:", test_inputs.shape)
print("Train labels shape:", train_labels.shape)
print("Test labels shape:", test_labels.shape)


# Train the perceptron
perceptron.train(train_inputs, train_labels, epochs=1000, learning_rate=0.1)
print("Training is successful!")

# Test the perceptron
correct_predictions = 0
for input_data, label in zip(test_inputs, test_labels):
    output = perceptron.forward_propagation(input_data)
    predicted_label = np.argmax(output)  # Get the index of the highest probability
    print(f"\nHere are the predicted probabilities: {output}, Predicted Labels: {predicted_label}, Actual Label: {label}")

    if predicted_label == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_labels)
print("Validation Accuracy:", accuracy)

# -------------------------------------- Model Training & Validation is completed ! ---------------------------------

# -------------------------------------- Lets test the Model with few images ! ---------------------------------

# Read the CSV file with the custom converter function
test_dataset = pd.read_csv("test_data_without_labels.csv", converters={"pixel_matrix": parse_pixel_matrix})
print("---------Test Results------------")
# Extract pixel matrices and labels from the dataset
pixel_matrices = test_dataset['pixel_matrix'].values
file_name = test_dataset['image_file_name'].values

for input_data, filename in zip(pixel_matrices, file_name):
    output = perceptron.forward_propagation(input_data)
    predicted_label = np.argmax(output)
    print(f"\nHere is the predicted label: {predicted_label} --> for filename: {filename}")
