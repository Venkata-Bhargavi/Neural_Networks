import numpy as np

class DepthwiseConvolution:
    def __init__(self, kernel):
        self.kernel = kernel

    def apply(self, image):
        output = np.zeros_like(image)
        for channel in range(image.shape[-1]):
            output[..., channel] = self._depthwise_convolve(image[..., channel], self.kernel)
        return output

    def _depthwise_convolve(self, image_channel, kernel):
        # Apply 2D convolution for a single channel
        return np.convolve(image_channel.ravel(), kernel.ravel(), mode='same').reshape(image_channel.shape)

class PointwiseConvolution:
    def __init__(self, kernel):
        self.kernel = kernel

    def apply(self, image):
        output = np.zeros_like(image)
        for i in range(self.kernel.shape[-1]):
            output += image * self.kernel[..., i]
        return output

