import numpy as np
import os
import matplotlib.image as mpimg
from skimage.transform import resize
from PIL import Image

PIC_COUNT = 200

def load_dataset(base_path, img_size=(1500, 1500)):
    """
    Loads a dataset of images, resizing them to a uniform size using matplotlib and skimage,
    ensuring all images have three color channels (RGB).

    :param base_path: Path to the directory with 'cat' and 'dog' subdirectories.
    :param img_size: Desired image size after resizing.
    :return: tuple of two numpy arrays: array of data and array of labels.
    """
    categories = ['cat', 'dog']  # Names of subdirectories for classes
    data = []  # To store image data
    labels = []  # To store class labels (0 for cats, 1 for dogs)

    for label, category in enumerate(categories):
        count = 0
        cat_dir = os.path.join(base_path, category)
        for img_name in os.listdir(cat_dir):
            if count > PIC_COUNT:
                break
            count += 1
            print(count)
            img_path = os.path.join(cat_dir, img_name)
            try:
                img = Image.open(img_path)  # Using PIL to open image to handle different formats
                img = img.convert('RGB')  # Convert to RGB
                img = np.array(img)  # Convert PIL image to numpy array
                img = resize(img, img_size, anti_aliasing=True, mode='reflect')  # Resize image

                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f'Error processing image {img_path}: {e}')

    # Convert lists to numpy arrays
    data = np.array(data, dtype='float32') / 255.0  # Normalize data
    labels = np.array(labels)

    return data, labels
