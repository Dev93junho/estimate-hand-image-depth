import os
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def load_data(image_dir, depth_map_dir):
    image_filenames = sorted(os.listdir(image_dir))
    depth_map_filenames = sorted(os.listdir(depth_map_dir))

    images = []
    depth_maps = []

    for img_filename, depth_map_filename in zip(image_filenames, depth_map_filenames):
        img_filepath = os.path.join(image_dir, img_filename)
        depth_map_filepath = os.path.join(depth_map_dir, depth_map_filename)

        img = imread(img_filepath)
        depth_map = imread(depth_map_filepath)

        images.append(img)
        depth_maps.append(depth_map)

    return np.array(images), np.array(depth_maps)


def preprocess_data(images, depth_maps, img_height, img_width, test_size=0.2):
    resized_images = []
    resized_depth_maps = []

    for img, depth_map in zip(images, depth_maps):
        # Resize the images and depth maps to the desired dimensions
        resized_img = resize(img, (img_height, img_width), preserve_range=True)
        resized_depth_map = resize(depth_map, (img_height, img_width), preserve_range=True)

        resized_images.append(resized_img)
        resized_depth_maps.append(resized_depth_map)

    resized_images = np.array(resized_images)
    resized_depth_maps = np.array(resized_depth_maps)

    # Normalize the images to the range [0, 1]
    normalized_images = resized_images / 255.0
    # Normalize the depth maps to the range [0, 1] if their maximum value is > 1
    normalized_depth_maps = resized_depth_maps / np.max(resized_depth_maps) if np.max(resized_depth_maps) > 1 else resized_depth_maps

    # Split the dataset into training and validation sets
    train_images, val_images, train_depth_maps, val_depth_maps = train_test_split(normalized_images, normalized_depth_maps, test_size=test_size, random_state=42)

    return train_images, val_images, train_depth_maps, val_depth_maps

def augmentation(images, depth_maps):
    pass

