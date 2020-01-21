import numpy as np


class NeuralNetwork:
    train_y: np.ndarray
    train_X: np.ndarray

    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    def training(self):
        pass
