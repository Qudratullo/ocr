print([2] * 4)

# import os
# import cv2
# import keras
# import numpy as np
# import tensorflow as tf
# from neural.dataset import load_data
# import matplotlib.pyplot as plt
#
# from prepare_image import convert_to_grayscale
# from word_segmentation import word_segmentation
# from prepare_image import resize_to_square
#
# model = keras.models.load_model('model/ocr_model', custom_objects={'softmax_v2': tf.nn.softmax})
#
# x_train, y_train, x_test, y_test = load_data("data", 10000, 1000)
# # Reshaping the array to 4-dims so that it can work with the Keras API
#
# print(x_test.shape)
#
# predict = model.predict(x_test[0].reshape(1, 32, 32, 1))
#
# print(predict.argmax(), y_test[0])
