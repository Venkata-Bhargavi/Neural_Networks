import pandas as pd

from helpers import generate_varied_handwritten_number_images_with_labels, \
    generate_test_images

# Generate varied handwritten-like images for numbers 0 to 9 with labels and pixel matrices stored in a DataFrame
dataset = pd.concat([generate_varied_handwritten_number_images_with_labels(num) for num in range(10)],
                    ignore_index=True)

dataset.to_csv("training_data.csv")

print("Varied Handwritten-like dataset with labels and pixel matrices generated successfully!")


# Generate test images for testing the trained perceptron
test_dataset = generate_test_images(num_images=5)
test_dataset.to_csv("test_data_without_labels.csv")


print("Test images generated successfully!")
