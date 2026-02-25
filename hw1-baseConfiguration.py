import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Feed-forward neural network with one hidden layer.
    - ReLU activation in the hidden layer
    - Linear (no activation) output layer
    - Bias units in both input and hidden layers
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network weights and biases from U(-0.01, 0.01).

        Args:
            input_size: number of input features
            hidden_size: number of hidden units
            output_size: number of output units
        """
        np.random.seed(0)

        # Weights (W) and biases (b) for hidden layer
        self.W1 = np.random.uniform(-0.01, 0.01, (input_size, hidden_size))
        self.b1 = np.random.uniform(-0.01, 0.01, (1, hidden_size))

        # Weights (W) and biases (b) for output layer
        self.W2 = np.random.uniform(-0.01, 0.01, (hidden_size, output_size))
        self.b2 = np.random.uniform(-0.01, 0.01, (1, output_size))

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: input data of shape (batch_size, input_size)

        Returns:
            output: linear output of shape (batch_size, output_size)
        """
        # Hidden layer: linear transform then ReLU
        self.z1 = X @ self.W1 + self.b1 #raw output of hidden layer before RELU is applied
        self.a1 = np.maximum(0, self.z1)  # ReLU activation

        # Output layer: linear transform only (no activation)
        self.z2 = self.a1 @ self.W2 + self.b2

        # Store input for use in backprop
        self.input = X

        return self.z2

    def backward(self, dL_dz2):
        """
        Backward pass — compute gradients for all weights and biases.

        Args:
            dL_dz2: gradient of the loss w.r.t. the output z2, shape (batch_size, output_size)
                    computed externally by the loss function
        """
        # Output layer gradients
        self.dW2 = self.a1.T @ dL_dz2          # gradient w.r.t. W2: (hidden_size, output_size)
        self.db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # sum across batch: (1, output_size)

        # Flow error back through W2 to reach the hidden layer
        self.da1 = dL_dz2 @ self.W2.T          # gradient w.r.t. a1: (batch_size, hidden_size)

        # Apply ReLU mask: neurons that were inactive (z1 <= 0) block the gradient
        self.dz1 = self.da1 * (self.z1 > 0)    # gradient w.r.t. z1: (batch_size, hidden_size)

        # Hidden layer gradients
        self.dW1 = self.input.T @ self.dz1     # gradient w.r.t. W1: (input_size, hidden_size)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)  # sum across batch: (1, hidden_size)


    def train(self, X, y, epochs, learning_rate, loss='cross_entropy'): 

        batch_size = 32
        loss_history = []

        for epoch in range(epochs):
            # shuffle the data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0  # reset loss accumulator each epoch

            # split into mini-batches
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                z2 = self.forward(X_batch)

                if loss == 'cross_entropy':
                    exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                    probs = exp / np.sum(exp, axis=1, keepdims=True)
                    dL_dz2 = probs.copy()
                    dL_dz2[range(len(X_batch)), y_batch] -= 1
                    dL_dz2 /= len(X_batch)
                    epoch_loss += -np.sum(np.log(probs[range(len(X_batch)), y_batch] + 1e-8))

                elif loss == 'mse':
                    dL_dz2 = (z2 - y_batch) / len(X_batch)
                    epoch_loss += np.sum((z2 - y_batch) ** 2)

                self.backward(dL_dz2)

                self.W1 -= learning_rate * self.dW1
                self.b1 -= learning_rate * self.db1
                self.W2 -= learning_rate * self.dW2
                self.b2 -= learning_rate * self.db2

            # record average loss for this epoch
            loss_history.append(epoch_loss / len(X))

        return loss_history

