import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W * W)
    grad = reg_strength * 2 * W
    
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    preds = predictions.copy()
    preds -= np.max(preds, axis=1, keepdims=True)
    return np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    n_samples = probs.shape[0]
    return np.mean(-np.log(probs[np.arange(n_samples), target_index]))


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probes = softmax(preds)
    loss = cross_entropy_loss(probes, target_index)
    
    dprediction = probes.copy()
    n_samples = probes.shape[0]
    dprediction[np.arange(n_samples), target_index] -= 1
    dprediction /= n_samples

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
# @np.vectorize
def ReLU(x):
    return x if x > 0 else 0

# def ReLU(X):
#     return (X + np.abs(X)) / 2


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.input = X.copy()
        return np.maximum(0, self.input)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_cur = self.input > 0
        d_result = d_out * d_cur
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Your final implementation shouldn't have any loops
        self.X = Param(X.copy())
        result = self.X.value.dot(self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_input = d_out.dot(self.W.value.T)
        
        grad_weights = self.X.value.T.dot(d_out)
        grad_bias = np.sum(d_out, axis=0).reshape(1, -1)
        
        self.W.grad += grad_weights
        self.B.grad += grad_bias

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
