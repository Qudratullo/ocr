import os
import numpy as np
import cv2

from prepare_image import convert_to_grayscale

IMAGE_FOLDER_NAME = 'images'


def text_split_to_lines(img):
    return []


def line_split_to_words(line):
    return []


def word_split_to_letters(word):
    return []


def recognize_letter(letter):
    return ''


def read_and_recognize_images():
    # read input images from 'in' directory
    img_files = os.listdir(f'{IMAGE_FOLDER_NAME}/')
    for (i, file_name) in enumerate(img_files):
        img = convert_to_grayscale(cv2.imread(f'{IMAGE_FOLDER_NAME}/{file_name}'))

        lines = text_split_to_lines(img)

        text = ''
        for line in lines:
            words = line_split_to_words(line)
            for word in words:
                letters = word_split_to_letters(word)
                for letter in letters:
                    text += recognize_letter(letter)
                text += ' '
            text += '\n'

        with open(f'{file_name}.txt') as file:
            file.write(text)


if __name__ == '__main__':
    read_and_recognize_images()
