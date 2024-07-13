from enum import Enum
import numpy as np
from scipy.special import softmax
from typing import List


class Activation(Enum):
    """
    The type of activation function for the layer.
    """
    relu = 'relu'
    sigmoid = 'sigmoid'
    softmax = 'softmax'
    linear = 'linear'


class Loss(Enum): 
    """
    The type of loss fn to use during training.
    """
    mse = 'mse'
    binary_cross_entropy = 'binary_cross_entropy'
    categorical_cross_entropy = 'categorical_cross_entropy'


class NeuralNetwork:
    """
    A simple neural network class.
    
    Attributes
    ----------
    input_size : int
        The size of the input layer.
    layers : List
        A list to store the layers of the network.
    weights : List
        A list to store the weights of the network.
    biases : List
        A list to store the biases of the network.
    """
    def __init__(self, input_size: int) -> None:
        """
        Initializes the network.

        Parameters
        ----------
        input_size : int
            The size of the input layer.
        """
        self.input_size = input_size
        self.layers : List[Layer] = []
        self.weights = []
        self.biases = []

    def __repr__(self):
        return str([f"Layer{i+1}{w.shape}" for i, w in enumerate(self.weights)])

    def add(self, layer):
        """
        Adds a layer to the neural network.

        Parameters
        ----------
        layer : Layer
            The new layer to be added to the network.
        """
        size = self.input_size if len(self.layers) == 0 else self.layers[-1].size
        self.layers.append(layer)
        self.weights.append(np.random.randn(size, layer.size) * np.sqrt(2 / size)) # He weight initialization
        self.biases.append(np.zeros((layer.size, )))

    def train(self, X, y, loss = Loss.binary_cross_entropy, epochs = 10, learning_rate = 0.001, batch_size=None):
        """
        Train the neural net.

        Parameters
        ----------
        X : numpy.ndarray
            The input matrix with shape (num_examples, num_features).
        Y : numpy.ndarray
            The true labels for each training example with shape (num_examples,).
        loss : Loss
            The loss function to be used for calculating the loss.
        epochs : int
            The number of training iterations.
        learning_rate : float
            The learning rate to be used during training.
        batch_size : int
            The number of examples per batch for training.
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        for epoch in range(epochs):
            if batch_size:
                i = 0
                while i < len(X):
                    X_batch = X[i: i + batch_size]
                    y_batch = y[i: i + batch_size]
                    A, Z = self.forward(X_batch)
                    dW, dB = self.backward(A, Z, y_batch, loss)
                    self._update_weights(dW, dB, learning_rate)
                    i += batch_size
            else:
                A, Z = self.forward(X)
                dW, dB = self.backward(A, Z, y)
                self._update_weights(dW, dB, learning_rate)

            loss = self._calculate_loss(A[-1], y_batch if batch_size else y, loss)
            print(f'Loss at step {epoch + 1}: {loss:.6f}')

    def predict(self, x):
        A, Z = self.forward(x)
        return A[-1]

    def forward(self, X: np.ndarray):
        """
        Feedforward to calculate the output of the network.

        Parameters
        ----------
        X : numpy.ndarray
            The inputs to the network.
        """
        a = X
        Z = []
        A = [ a ]
        for w, b, layer in zip(self.weights, self.biases, self.layers):
            z = np.array(a.dot(w) + b)
            Z.append(z) 
            a = np.array(layer.activate(z))
            A.append(a)

        return A, Z

    def backward(self, A: np.ndarray, Z: np.ndarray, y: np.ndarray, loss=Loss.binary_cross_entropy):
        """
        Backpropagation to calculate the gradients.

        Parameters
        ----------
        A: np.ndarray
            Activation outputs at each layer.
        Z: np.ndarray
            Regular outputs at each layer. 
        y: np.ndarray
            Ground truth labels.
        """
        m = y.shape[0] 
        num_layers = len(self.layers)

        dCdZ = [0.0] * num_layers
        dCdZ[-1] = self.derive_loss(A[-1], y, loss) * self.layers[-1].derivative(Z[-1])
        for i in reversed(range(num_layers - 1)):
            dA = dCdZ[i+1].dot(self.weights[i+1].T)
            dCdZ[i] = dA * self.layers[i].derivative(Z[i])

        dW = []
        dB = []
        for i, dZ in enumerate(dCdZ):
            dW.append(A[i].T.dot(dZ) / m)
            dB.append(np.sum(dZ, axis=0) / m)

        return dW, dB

    def _update_weights(self, dW, dB, learning_rate): 
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * dW[i]
            self.biases[i] = self.biases[i] - learning_rate * dB[i]

    def _calculate_loss(self, a, y, loss=Loss.binary_cross_entropy): 
        """
        Calculates loss function.

        Parameters
        ----------
        a: numpy.ndarray
            Predictions from the forward pass.
        y: numpy.ndarray
            Ground truth labels.
        loss: Loss
            Loss function.
        """
        epsilon = 1e-15 
        if loss == Loss.binary_cross_entropy:
            a = np.clip(a, epsilon, 1 - epsilon)  # avoid log(0)
            return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
        elif loss == Loss.categorical_cross_entropy: 
            a = np.clip(a, epsilon, 1 - epsilon) 
            return -np.sum(y * np.log(a)) / y.shape[0]
        else:
            return np.mean((y - a)**2)

    def derive_loss(self, a, y, loss=Loss.binary_cross_entropy):
        epsilon = 1e-15
        if loss == Loss.binary_cross_entropy:
            a = np.clip(a, epsilon, 1 - epsilon)
            return (a - y) / (a * (1 - a))
        elif loss == Loss.categorical_cross_entropy:
            a = np.clip(a, epsilon, 1 - epsilon)
            return a - y
        else:
            return a - y

    @staticmethod
    def one_hot_encode(y, labels: List):
        y_encoded = []
        for out in y: 
            one_hot = [0.0] * len(labels)
            idx = labels.index(out)
            one_hot[idx] = 1.0
            y_encoded.append(one_hot)
        
        return np.array(y_encoded)

    @staticmethod
    def one_hot_decode(y, labels : List):
        y_decoded = []
        for y_pred in y: 
            label = np.argmax(y_pred)
            category = labels[label]
            y_decoded.append(category)

        return np.array(y_decoded)
    
    @staticmethod
    def one_hot_normalization(y_pred): 
        out = []
        for i, y in enumerate(y_pred):
            one_hot = [0.0] * len(y)
            idx = np.argmax(y)
            one_hot[idx] = 1.0
            out.append(one_hot)
        
        return np.array(out)
    
    @staticmethod
    def accuracy(y_pred, y):
        return np.sum(y_pred == y)        


class Layer:
    def __init__(
            self,
            size : int,
            activation = Activation.relu        
        ) -> None:
            self.size = size
            self.activation = activation

    def activate(self, x): 
         if self.activation == Activation.relu:
              return np.maximum(0, x)
         elif self.activation == Activation.sigmoid:
            return 1 / (1 + np.exp(-x))
         elif self.activation == Activation.softmax: 
            x = x - np.max(x, axis=-1, keepdims=True)
            return softmax(x, axis=-1)
         else:
             return x
    
    def derivative(self, x):
         if self.activation == Activation.relu:
              return np.where(x > 0, 1, 0)
         elif self.activation == Activation.sigmoid:
            s = self.activate(x)
            return (s * (1 - s))
         elif self.activation == Activation.softmax: 
             s = softmax(x)
             return s * (1 - s)
         else:
             return np.zeros(x.shape)
