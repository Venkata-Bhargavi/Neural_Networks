import numpy as np
import matplotlib.pyplot as plt

from convolution_classes import DepthwiseConvolution,PointwiseConvolution

def main():
    # Example input image and kernel
    image = np.array([[[120, 150, 180], [100, 120, 140], [80, 100, 120]],
                      [[200, 180, 160], [180, 160, 140], [160, 140, 120]],
                      [[50, 100, 150], [70, 120, 170], [90, 140, 190]]])

    kernel_depthwise = np.array([[1, 0, -1],
                                  [-1, 0, 1],
                                  [1, 0, -1]])

    kernel_pointwise = np.array([[[1], [0], [1]],
                                  [[0], [1], [0]],
                                  [[-1], [0], [-1]]])

    # External flag to indicate the type of convolution (depthwise or pointwise)
    convolution_type = 'depthwise'  # Change to 'pointwise' for pointwise convolution

    if convolution_type == 'depthwise':
        depthwise_convolution = DepthwiseConvolution(kernel_depthwise)
        convolved_image = depthwise_convolution.apply(image)
    elif convolution_type == 'pointwise':
        pointwise_convolution = PointwiseConvolution(kernel_pointwise)
        convolved_image = pointwise_convolution.apply(image)
    else:
        raise ValueError("Invalid convolution type. Choose 'depthwise' or 'pointwise'.")

    # Visualize original and convolved images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.transpose(1, 2, 0), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(convolved_image.transpose(1, 2, 0), cmap='gray')
    axes[1].set_title('Convolved Image')
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()