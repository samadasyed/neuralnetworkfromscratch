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

# ── Iris Experiment ──────────────────────────────────────────────────────────

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Encode string labels to integers
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['species'] = df['species'].map(label_map)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = df['species'].values

# 80/20 train/test split
np.random.seed(0)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Helper: predict class labels
def predict(nn, X):
    z2 = nn.forward(X)
    return np.argmax(z2, axis=1)

# ── 1a: 5 hidden units, 10 epochs, 4 learning rates ──────────────────────────
learning_rates = [1, 0.1, 1e-2, 1e-3, 1e-8]

plt.figure()
for lr in learning_rates:
    nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
    loss_history = nn.train(X_train, y_train, epochs=10, learning_rate=lr)
    plt.plot(loss_history, label=f"LR={lr}")

plt.title("Iris Training Loss by Learning Rate (5 hidden units, 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Average Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── 1c: average test loss for each LR ────────────────────────────────────────
print("1c. Test losses by learning rate:")
for lr in learning_rates:
    nn = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
    nn.train(X_train, y_train, epochs=10, learning_rate=lr)
    z2 = nn.forward(X_test)
    exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    test_loss = -np.mean(np.log(probs[range(len(X_test)), y_test] + 1e-8))
    print(f"  LR={lr}: test loss = {test_loss:.4f}")

# ── 1d: LR=1e-2, 10 epochs, 4 hidden sizes ───────────────────────────────────
hidden_sizes = [2, 8, 16, 32]

plt.figure()
for h in hidden_sizes:
    nn = NeuralNetwork(input_size=4, hidden_size=h, output_size=3)
    loss_history = nn.train(X_train, y_train, epochs=10, learning_rate=1e-2)
    plt.plot(loss_history, label=f"Hidden={h}")

plt.title("Iris Training Loss by Hidden Size (LR=1e-2, 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Average Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── 1e: test loss and accuracy for each hidden size ───────────────────────────
print("1e. Test loss and accuracy by hidden size:")
for h in hidden_sizes:
    nn = NeuralNetwork(input_size=4, hidden_size=h, output_size=3)
    nn.train(X_train, y_train, epochs=10, learning_rate=1e-2)
    z2 = nn.forward(X_test)
    exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    test_loss = -np.mean(np.log(probs[range(len(X_test)), y_test] + 1e-8))
    accuracy = np.mean(predict(nn, X_test) == y_test)
    print(f"  Hidden={h}: test loss = {test_loss:.4f}, accuracy = {accuracy * 100:.1f}%")

# ── California Housing Experiment ─────────────────────────────────────────────

# Load dataset
housing_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
housing_df = pd.read_csv(housing_url)

# Drop rows with missing values
housing_df = housing_df.dropna()

# One-hot encode the categorical column
housing_df = pd.get_dummies(housing_df, columns=['ocean_proximity'])

# Separate features and target
target = 'median_house_value'
feature_cols = [c for c in housing_df.columns if c != target]
X_housing = housing_df[feature_cols].values.astype(float)
y_housing = housing_df[target].values.astype(float).reshape(-1, 1)

# Standardize input features to zero mean and variance 1
X_mean = X_housing.mean(axis=0)
X_std = X_housing.std(axis=0) + 1e-8  # avoid division by zero
X_housing = (X_housing - X_mean) / X_std

# Standardize target
y_mean = y_housing.mean()
y_std = y_housing.std()
y_housing = (y_housing - y_mean) / y_std

# 80/20 train/test split
np.random.seed(0)
indices = np.random.permutation(len(X_housing))
split = int(0.8 * len(X_housing))
X_htrain, X_htest = X_housing[indices[:split]], X_housing[indices[split:]]
y_htrain, y_htest = y_housing[indices[:split]], y_housing[indices[split:]]

input_size = X_housing.shape[1]  # number of features after one-hot encoding

# ── 2a: 5 hidden units, 10 epochs, learning rates ────────────────────────────
learning_rates = [1, 0.1, 1e-2, 1e-3, 1e-8]

plt.figure()
for lr in learning_rates:
    nn = NeuralNetwork(input_size=input_size, hidden_size=5, output_size=1)
    loss_history = nn.train(X_htrain, y_htrain, epochs=10, learning_rate=lr, loss='mse')
    plt.plot(loss_history, label=f"LR={lr}")

plt.title("Housing Training Loss by Learning Rate (5 hidden units, 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── 2c: average test MSE for each LR ─────────────────────────────────────────
print("2c. Test MSE by learning rate:")
for lr in learning_rates:
    nn = NeuralNetwork(input_size=input_size, hidden_size=5, output_size=1)
    nn.train(X_htrain, y_htrain, epochs=10, learning_rate=lr, loss='mse')
    z2 = nn.forward(X_htest)
    test_loss = np.mean((z2 - y_htest) ** 2)
    print(f"  LR={lr}: test MSE = {test_loss:.4f}")

# ── 2d: LR=1e-2, 10 epochs, 4 hidden sizes ───────────────────────────────────
hidden_sizes = [2, 8, 16, 32]

plt.figure()
for h in hidden_sizes:
    nn = NeuralNetwork(input_size=input_size, hidden_size=h, output_size=1)
    loss_history = nn.train(X_htrain, y_htrain, epochs=10, learning_rate=1e-2, loss='mse')
    plt.plot(loss_history, label=f"Hidden={h}")

plt.title("Housing Training Loss by Hidden Size (LR=1e-2, 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Average MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── 2e: test MSE for each hidden size ────────────────────────────────────────
print("2e. Test MSE by hidden size:")
for h in hidden_sizes:
    nn = NeuralNetwork(input_size=input_size, hidden_size=h, output_size=1)
    nn.train(X_htrain, y_htrain, epochs=10, learning_rate=1e-2, loss='mse')
    z2 = nn.forward(X_htest)
    test_loss = np.mean((z2 - y_htest) ** 2)
    print(f"  Hidden={h}: test MSE = {test_loss:.4f}")

# ── MNIST Experiment ──────────────────────────────────────────────────────────
from readinginMNIST import load_images, load_labels

# Load MNIST data
X_mnist_train = load_images("train-images-idx3-ubyte")
y_mnist_train = load_labels("train-labels-idx1-ubyte")
X_mnist_test = load_images("t10k-images-idx3-ubyte")
y_mnist_test = load_labels("t10k-labels-idx1-ubyte")

# Flatten 28x28 images to 784-element vectors
X_mnist_train = X_mnist_train.reshape(X_mnist_train.shape[0], -1).astype(float) / 255.0
X_mnist_test = X_mnist_test.reshape(X_mnist_test.shape[0], -1).astype(float) / 255.0

# Use subset of 20,000 training examples to keep runtime manageable
X_mnist_train = X_mnist_train[:20000]
y_mnist_train = y_mnist_train[:20000]

# ── 3a: 5 hidden units, 10 epochs, learning rates ────────────────────────────
learning_rates = [1, 0.1, 1e-2, 1e-3, 1e-8]

plt.figure()
for lr in learning_rates:
    nn = NeuralNetwork(input_size=784, hidden_size=5, output_size=10)
    loss_history = nn.train(X_mnist_train, y_mnist_train, epochs=10, learning_rate=lr)
    plt.plot(loss_history, label=f"LR={lr}")

plt.title("MNIST Training Loss by Learning Rate (5 hidden units, 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Average Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── 3c: average test loss for each LR ────────────────────────────────────────
print("3c. Test losses by learning rate:")
for lr in learning_rates:
    nn = NeuralNetwork(input_size=784, hidden_size=5, output_size=10)
    nn.train(X_mnist_train, y_mnist_train, epochs=10, learning_rate=lr)
    z2 = nn.forward(X_mnist_test)
    exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    test_loss = -np.mean(np.log(probs[range(len(X_mnist_test)), y_mnist_test] + 1e-8))
    print(f"  LR={lr}: test loss = {test_loss:.4f}")

# ── 3d: LR=1e-2, 10 epochs, 4 hidden sizes ───────────────────────────────────
hidden_sizes = [2, 8, 16, 32]

plt.figure()
for h in hidden_sizes:
    nn = NeuralNetwork(input_size=784, hidden_size=h, output_size=10)
    loss_history = nn.train(X_mnist_train, y_mnist_train, epochs=10, learning_rate=1e-2)
    plt.plot(loss_history, label=f"Hidden={h}")

plt.title("MNIST Training Loss by Hidden Size (LR=1e-2, 10 epochs)")
plt.xlabel("Epoch")
plt.ylabel("Average Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ── 3e: test loss and accuracy for each hidden size ───────────────────────────
print("3e. Test loss and accuracy by hidden size:")
for h in hidden_sizes:
    nn = NeuralNetwork(input_size=784, hidden_size=h, output_size=10)
    nn.train(X_mnist_train, y_mnist_train, epochs=10, learning_rate=1e-2)
    z2 = nn.forward(X_mnist_test)
    exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    test_loss = -np.mean(np.log(probs[range(len(X_mnist_test)), y_mnist_test] + 1e-8))
    accuracy = np.mean(np.argmax(z2, axis=1) == y_mnist_test)
    print(f"  Hidden={h}: test loss = {test_loss:.4f}, accuracy = {accuracy * 100:.1f}%")
