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

        # Weights and biases for hidden layer
        self.W1 = np.random.uniform(-0.01, 0.01, (input_size, hidden_size))
        self.b1 = np.random.uniform(-0.01, 0.01, (1, hidden_size))

        # Weights and biases for output layer
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
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation

        # Output layer: linear transform only (no activation)
        self.z2 = self.a1 @ self.W2 + self.b2

        # Store input for use in backprop
        self.input = X

        return self.z2
