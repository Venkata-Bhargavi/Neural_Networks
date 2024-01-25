from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
import cv2
import pandas as pd


## updated

class Perceptron:
    def __init__(self, input_size, activation_function="sigmoid"):
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation_function = activation_function

    def initialize_parameters(self):
        self.weights = np.random.randn(self.input_size)
        self.bias = 0 #np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid(x)
        else:
            raise NotImplementedError("Activation function not implemented.")

    def activation_derivative(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid_derivative(x)
        else:
            raise NotImplementedError("Activation function derivative not implemented.")

    def forward_propagation(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        output = self.activation(weighted_sum)
        return output

    def calculate_loss(self, predicted, target):
        # Mean Squared Error (MSE) loss
        return np.mean((predicted - target) ** 2)

    def calculate_gradient(self, inputs, output, target):
        error = target - output
        gradient = self.activation_derivative(output)
        delta = error * gradient
        d_weights = np.dot(inputs.T, delta)
        d_bias = np.sum(delta)
        return d_weights, d_bias

    def update_parameters(self, d_weights, d_bias, learning_rate):
        self.weights += learning_rate * d_weights
        self.bias += learning_rate * d_bias

    def train(self, inputs, targets, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            total_loss = 0

            for input_data, target in zip(inputs, targets):
                # Forward propagation
                output = self.forward_propagation(input_data)

                # Calculate loss
                loss = self.calculate_loss(output, target)
                total_loss += loss

                # Backward propagation
                d_weights, d_bias = self.calculate_gradient(input_data, output, target)
                self.update_parameters(d_weights, d_bias, learning_rate)

            # Print average loss for each epoch
            if epoch % 5 == 0:
                average_loss = total_loss / len(inputs)
                print(f"Epoch {epoch}: Average Loss = {average_loss}")

# Function to create variations of handwritten numbers with noise, shift, blur, and rotation
def generate_varied_handwritten_number_images_with_labels(number, num_variations=10):
    data = []

    for i in range(num_variations):
        image = Image.new("L", (20, 20), color=255)  # Create a white background
        draw = ImageDraw.Draw(image)

        # You can customize the font, position, and other parameters
        draw.text((5, 5), str(number), fill=0)  # Draw the number in black

        # Add some noise to make it look handwritten
        array_image = np.array(image)
        noise = np.random.normal(0, 20, array_image.shape)
        handwritten_image = np.clip(array_image + noise, 0, 255).astype(np.uint8)

        # Shift pixels randomly
        shift_x, shift_y = np.random.randint(-2, 3, 2)
        handwritten_image = np.roll(handwritten_image, shift_x, axis=1)
        handwritten_image = np.roll(handwritten_image, shift_y, axis=0)

        # Apply blur
        # handwritten_image = cv2.GaussianBlur(handwritten_image, (3, 3), 0)

        # Rotate image
        rotation_angle = np.random.uniform(-10, 10)
        handwritten_image_rotated = Image.fromarray(handwritten_image)
        handwritten_image_rotated = handwritten_image_rotated.rotate(rotation_angle, resample=Image.BICUBIC,
                                                                     fillcolor=255)

        # Convert image to DataFrame
        pixel_matrix = handwritten_image_rotated.getdata()
        pixel_matrix = np.array(pixel_matrix).reshape((20, 20))

        # Append label and pixel matrix to the data list
        image_filename = f"{number}_{i}"
        data.append({'image_file_name': image_filename, 'pixel_matrix': pixel_matrix.flatten(), 'label': number})

        # Save the varied handwritten-like image
        handwritten_image_rotated.save(f"dataset_handwritten_variations/{number}_{i}.png")

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    return df




# Function to create test images without labels
def generate_test_images(num_images=5):
    test_data = []

    # Create directory to store test images if it doesn't exist
    if not os.path.exists("test_images"):
        os.makedirs("test_images")

    for i in range(num_images):
        image = Image.new("L", (20, 20), color=255)  # Create a white background
        draw = ImageDraw.Draw(image)

        # You can customize the font, position, and other parameters
        draw.text((5, 5), str(np.random.randint(0, 10)), fill=0)  # Draw a random number (0 to 9) in black

        # Add some noise to make it look handwritten
        array_image = np.array(image)
        noise = np.random.normal(0, 20, array_image.shape)
        handwritten_image = np.clip(array_image + noise, 0, 255).astype(np.uint8)

        # Shift pixels randomly
        shift_x, shift_y = np.random.randint(-2, 3, 2)
        handwritten_image = np.roll(handwritten_image, shift_x, axis=1)
        handwritten_image = np.roll(handwritten_image, shift_y, axis=0)

        # Apply blur
        # handwritten_image = cv2.GaussianBlur(handwritten_image, (3, 3), 0)

        # Rotate image
        rotation_angle = np.random.uniform(-10, 10)
        handwritten_image_rotated = Image.fromarray(handwritten_image)
        handwritten_image_rotated = handwritten_image_rotated.rotate(rotation_angle, resample=Image.BICUBIC,
                                                                     fillcolor=255)

        # Convert image to DataFrame
        pixel_matrix = handwritten_image_rotated.getdata()
        pixel_matrix = np.array(pixel_matrix).reshape((20, 20))

        # Append pixel matrix to the test data list
        image_filename = f"test_image_{i}"
        test_data.append({'image_file_name': image_filename, 'pixel_matrix': pixel_matrix.flatten()})

        # Save the test image
        handwritten_image_rotated.save(f"test_images/{image_filename}.png")

    # Create DataFrame from the collected test data
    df = pd.DataFrame(test_data)
    return df


