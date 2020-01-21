import cv2
import numpy as np


def prepare_image(image_path):
    img = convert_to_grayscale(cv2.imread(image_path))
    height, width = img.shape
    images = []

    if width > 32:
        images.append(resize_to_square(img, 32))
        return images

    for i in range(33 - width):
        images.append(get_square_images(img, 32, i))

    return images


def convert_to_grayscale(img):
    """convert given image to grayscale image (if needed) and resize to desired height"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def get_square_images(img, square_size, x_pos):
    height, width = img.shape

    mask = np.ones((square_size, square_size), dtype="uint8") * 255

    mask[:height, x_pos:x_pos + width] = img[:height, :width]
    mask = cv2.resize(mask, (square_size, square_size), interpolation=cv2.INTER_AREA)

    return mask


def resize_to_square(img, square_size):
    height, width = img.shape
    if height > width:
        differ = height
    else:
        differ = width

    mask = np.ones((differ, differ), dtype="uint8") * 255
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos:y_pos + height, x_pos:x_pos + width] = img[0:height, 0:width]
    mask = cv2.resize(mask, (square_size, square_size), interpolation=cv2.INTER_AREA)

    return mask

#[4.982544872212406, 0.828000009059906]