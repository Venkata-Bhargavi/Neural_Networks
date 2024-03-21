import numpy as np

class Convolution:
    def __init__(self):
        pass

    def apply_convolution(self, image, kernel):
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape
        output_height = image_height - kernel_height + 1
        output_width = image_width - kernel_width + 1
        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)

        return output

def main():
    # Define the original image (6x6) and the filter (3x3)
    image = np.array([
        [100, 120, 110, 90, 80, 70],
        [105, 125, 115, 95, 85, 75],
        [110, 130, 120, 100, 90, 80],
        [115, 135, 125, 105, 95, 85],
        [120, 140, 130, 110, 100, 90],
        [125, 145, 135, 115, 105, 95]
    ])

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Create Convolution object
    conv = Convolution()

    # Perform convolution
    output = conv.apply_convolution(image, kernel)

    # Output the result
    print("Original Image:\n", image)
    print("\nFilter:\n", kernel)
    print("\nOutput Image after Convolution:\n", output)

if __name__ == "__main__":
    main()
