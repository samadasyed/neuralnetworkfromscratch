import numpy as np
import struct

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)
        return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
        
X_train = load_images("train-images-idx3-ubyte")
y_train = load_labels("train-labels-idx1-ubyte")
X_test = load_images("t10k-images-idx3-ubyte")
y_test = load_labels("t10k-labels-idx1-ubyte")
# Flatten
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# Normalize to [0,1]
X_train = X_train.astype(float) / 255.0
X_test = X_test.astype(float) / 255.0