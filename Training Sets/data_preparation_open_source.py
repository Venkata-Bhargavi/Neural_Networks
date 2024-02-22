import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from PIL import Image

class MNISTDatasetGenerator:
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        # Load the MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return train_images, train_labels, test_images, test_labels

    def split_dataset(self, train_images, train_labels, test_images, test_labels):
        # Combine train and test data to create full dataset
        full_images = np.concatenate([train_images, test_images], axis=0)
        full_labels = np.concatenate([train_labels, test_labels], axis=0)

        # Split the dataset into train, validation, and test sets
        train_images, val_test_images, train_labels, val_test_labels = train_test_split(
            full_images, full_labels, test_size=self.test_size, random_state=self.random_state)
        val_images, test_images, val_labels, test_labels = train_test_split(
            val_test_images, val_test_labels, test_size=0.5, random_state=self.random_state)

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def create_folders(self):
        # Create folders to store the datasets
        os.makedirs('data_open_source/train', exist_ok=True)
        os.makedirs('data_open_source/val', exist_ok=True)
        os.makedirs('data_open_source/test', exist_ok=True)

    def save_images(self, images, labels, folder):
        # Save images to folders
        for i, (image, label) in enumerate(zip(images, labels)):
            image_folder = os.path.join(folder, str(label))
            os.makedirs(image_folder, exist_ok=True)
            image_path = os.path.join(image_folder, f"{i}.png")
            image = Image.fromarray(image)
            image.save(image_path)

    def generate_and_store_data(self):
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_data()

        # Split dataset
        train_images, train_labels, val_images, val_labels, test_images, test_labels = self.split_dataset(
            train_images, train_labels, test_images, test_labels)

        # Create folders
        self.create_folders()

        # Save images
        self.save_images(train_images, train_labels, 'data/train')
        self.save_images(val_images, val_labels, 'data/val')
        self.save_images(test_images, test_labels, 'data/test')

        print("Data generation and storage completed.")


# Example usage:
mnist_generator = MNISTDatasetGenerator()
mnist_generator.generate_and_store_data()
