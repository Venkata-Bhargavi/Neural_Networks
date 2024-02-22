import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


class HandwrittenImageGenerator:
    def __init__(self, num_variations=10, image_size=(20, 20), rotation_range=(-10, 10), shift_range=(-2, 3)):
        self.num_variations = num_variations
        self.image_size = image_size
        self.rotation_range = rotation_range
        self.shift_range = shift_range

    def generate_varied_handwritten_number_images_with_labels(self, number):
        '''
        function to generate hand written images with 1 different variations
        :param number: required number
        :return: dataframe
        '''
        data = []

        for i in range(self.num_variations):
            image = Image.new("L", self.image_size, color=255)  # Create a white background
            draw = ImageDraw.Draw(image)

            # You can customize the font, position, and other parameters
            draw.text((10, 10), str(number), fill=0)  # Draw the number in black

            # Shift pixels randomly
            shift_x, shift_y = np.random.randint(*self.shift_range, size=2)
            handwritten_image = np.array(image)
            handwritten_image = np.roll(handwritten_image, shift_x, axis=1)
            handwritten_image = np.roll(handwritten_image, shift_y, axis=0)

            # Rotate image
            rotation_angle = np.random.uniform(*self.rotation_range)
            handwritten_image_rotated = Image.fromarray(handwritten_image)
            handwritten_image_rotated = handwritten_image_rotated.rotate(rotation_angle, resample=Image.BICUBIC,
                                                                         fillcolor=255)

            # Convert image to DataFrame
            pixel_matrix = handwritten_image_rotated.getdata()
            pixel_matrix = np.array(pixel_matrix).reshape(self.image_size)

            # Append label and pixel matrix to the data list
            image_filename = f"{number}_{i}"
            data.append({'image_file_name': image_filename, 'pixel_matrix': pixel_matrix.flatten(), 'label': number})

            # Save the varied handwritten-like image
            handwritten_image_rotated.save(f"dataset_handwritten_variations/{number}_{i}.png")

        # Create DataFrame from the collected data
        df = pd.DataFrame(data)
        return df

    def generate_dataset(self):
        dataset = pd.concat([self.generate_varied_handwritten_number_images_with_labels(num) for num in range(10)],
                            ignore_index=True)
        return dataset


# Example usage:
generator = HandwrittenImageGenerator()
dataset = generator.generate_dataset()
