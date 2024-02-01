import numpy as np

from helpers import NeuralNetwork

X = np.array([[2, 3], [1, 2], [3, 2], [2, 1]]) # 4 data points with 2 features
y = np.array([[0], [0], [1], [1]]) # (4,1) ------ 4 target outputs

# Initialize and train the neural network
model = NeuralNetwork(activation='sigmoid')  # You can choose 'sigmoid' or 'relu' for the activation function
model.train(X, y, epochs=1000, learning_rate=0.1)
