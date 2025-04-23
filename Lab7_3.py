#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import load_iris


# Define the SVM loss function
def svm_loss(w, b, X, y, C):
    """
    Compute the SVM loss function.

    Inputs:
    - w: A numpy array of shape (D,) containing weights.
    - b: A scalar bias term.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = -1 or 1
      corresponds to the class of X[i].
    - C: A float hyperparameter for the SVM.

    Returns a tuple of:
    - loss: SVM loss.
    - dW: Gradient of the loss with respect to weights w.
    - db: Gradient of the loss with respect to bias b.
    """
    N, D = X.shape
    loss = 0.0
    dW = np.zeros_like(w)
    db = 0.0
    for i in range(N):
        scores = np.dot(X[i], w) + b
        margin = y[i] * scores
        if margin < 1:
            loss += 1 - margin
            dW += -y[i] * X[i]
            db += -y[i]
    loss = C * loss + 0.5 * np.sum(w ** 2)
    dW = C * dW + w
    db = C * db
    return loss, dW, db


iris = load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]
y = np.where(y == 0, -1, 1)

C = 1.0
w = np.zeros(X.shape[1])
b = 0.0

num_epochs = 1000
learning_rate = 0.01
for epoch in range(num_epochs):
    # Shuffle the training data
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Compute the loss and gradients
    loss, dW, db = svm_loss(w, b, X_shuffled, y_shuffled, C)

    # Update the weights and bias using gradient descent
    w -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

# Evaluate the SVM
y_pred = np.sign(np.dot(X, w) + b)
accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.4f}")

# Visualization for all feature pairs
n_classes = 2
colors = 'bwr'  # blue for -1, red for 1
CMAP = colors
plot_step = 0.02

fig = plt.figure(1, figsize=(18, 9))

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X_pair = X[:, pair]

    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X_pair[:, 0].min() - 0.5, X_pair[:, 0].max() + 0.5
    y_min, y_max = X_pair[:, 1].min() - 0.5, X_pair[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # Predict for each point in the meshgrid
    Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w[pair]) + b)
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=CMAP, alpha=0.5)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    # Plot the training points
    for i, color in zip([-1, 1], ['b', 'r']):
        idx = np.where(y == i)
        plt.scatter(X_pair[idx, 0], X_pair[idx, 1], c=color, edgecolor='black',
                    lw=1, label=f"Class {i}", cmap=CMAP)

    plt.axis("tight")
y_pred = np.sign(np.dot(X, w) + b)
accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.4f}")
plt.suptitle("SVM decision boundaries for Iris dataset (binary classification)")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()