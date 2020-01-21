import os
import cv2
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import preprocessor as pp

from prepare_image import convert_to_grayscale
from word_segmentation import word_segmentation
from prepare_image import resize_to_square

model = keras.models.load_model('model/ocr_model', custom_objects={'softmax_v2': tf.nn.softmax})
MIN_AREA_OF_VALID_LETTER = 150
MIN_WIDTH_LETTER = 7
plot_index = 0


def recognize_letter(img):
    image = img.copy()
    image = resize_to_square(image, 32)
    plt.imshow(image.astype(np.uint8))
    get = model.predict(image.reshape(1, 32, 32, 1))
    return chr(get.argmax())


def is_empty(matrix):
    matrixA = 255 - matrix.copy()
    return np.sum(matrixA) - np.max(matrixA) < MIN_AREA_OF_VALID_LETTER


def split_to_letters(image_path, f):
    org_img = cv2.imread(f'{image_path}/{f}.png')
    img = convert_to_grayscale(org_img)
    h, w = img.shape

    if not os.path.exists(f'{image_path}/{f}'):
        os.mkdir(f'{image_path}/{f}')

    word = ''
    last = 0
    for j0 in range(w):
        if is_empty(img[:, j0:j0 + 1]) and not is_empty(img[:, last:j0 + 1]) and j0 - last > MIN_WIDTH_LETTER:
            word += recognize_letter(img[:, last:j0 + 1])
            cv2.imwrite(f'{image_path}/{f}/{j0}.png', img[:, last:j0 + 1])
            cv2.rectangle(img, (last, 0), (j0, h), 0, 1)
            last = j0

    if not is_empty(img[:, last:w]) and w - last > MIN_WIDTH_LETTER:
        word += recognize_letter(img[:, last:w])
        cv2.imwrite(f'{image_path}/{f}/{w}.png', img[:, last:w])
        cv2.rectangle(img, (last, 0), (w, h), 0, 1)

    cv2.imwrite(f'{image_path}/{f}/summary.png', img)
    return word


def split_to_words(image_path, f):
    org_img = cv2.imread(f'{image_path}/{f}.png')
    img = convert_to_grayscale(org_img)
    res = word_segmentation(img, kernel_size=15, sigma=11, theta=5, min_area=50)

    if not os.path.exists(f'{image_path}/{f}'):
        os.mkdir(f'{image_path}/{f}')

    line = ''
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w

        if is_empty(wordImg):
            continue

        (x, y, w, h) = wordBox
        cv2.imwrite(f'{image_path}/{f}/{j}.png', org_img[max(y - 2, 0):min(y + h + 2, img.shape[1]),
                                                 max(x - 2, 0):min(x + w + 2, img.shape[1])])  # save word
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image

        line += split_to_letters(f'{image_path}/{f}', j)
        line += ' '

    # output summary image with bounding boxes around words
    cv2.imwrite(f'{image_path}/{f}/summary.png', img)
    return line


def split_to_lines(image_path, f):
    org_img = cv2.imread(f'{image_path}/{f}.png')
    img = convert_to_grayscale(org_img)
    res = word_segmentation(img, kernel_size=55, sigma=14, theta=7, min_area=2000, sort_second=True)

    if not os.path.exists(f'{image_path}/{f}'):
        os.mkdir(f'{image_path}/{f}')

    text = ''
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite(f'{image_path}/{f}/{j}.png', org_img[y:y + h, x:x + w])  # save word
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)  # draw bounding box in summary image
        text += split_to_words(f'{image_path}/{f}', j)
        text += '\n'

    # output summary image with bounding boxes around words
    cv2.imwrite(f'{image_path}/{f}/summary.png', img)
    return pp.get_string(f'{image_path}/{f}.png')


if __name__ == '__main__':
    # print(split_to_lines('images', 'russian'))
    print(split_to_lines('images', 'learning_dart'))
