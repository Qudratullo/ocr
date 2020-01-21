import numpy as np

from neural.layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for layer in self.layers:
            for param in layer.params():
                layer.params()[param].grad = np.zeros_like(layer.params()[param].grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        X_cur = X.copy()
        for layer in self.layers:
            X_cur = layer.forward(X_cur)
        
        loss, grad = softmax_with_cross_entropy(X_cur, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for layer in reversed(self.layers):
            for param, value in layer.params().items():
                loss_reg, grad_reg = l2_regularization(value.value, self.reg)
                loss += loss_reg
                layer.params()[param].grad += grad_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        pred = X.copy()
        for layer in self.layers:
            pred = layer.forward(pred)
        pred = np.argmax(pred, axis=1)
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for i, layer in enumerate(self.layers):
            for param in layer.params():
                result[f'layer{i}_{param}'] = layer.params()[param]

        return result
