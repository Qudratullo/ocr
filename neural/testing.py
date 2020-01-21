import keras
import tensorflow as tf

from neural.dataset import load_data


def main():
    x_train, y_train, x_test, y_test = load_data("../data", 10000, 1000)
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    model = keras.models.load_model('ocr_model', custom_objects={'softmax_v2': tf.nn.softmax})

    scores = model.evaluate(x_test, y_test)

    print(scores)


if __name__ == '__main__':
    main()
