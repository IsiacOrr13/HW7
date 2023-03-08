"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import (logreg, utils)
import numpy as np

# (you will probably need to import more things here)

X_t  = np.array([[1, 2, 3,1, 2, 3], [1, 2, 2,1, 2, 3], [2, 1, 4,1, 2, 3], [1, 2, 5,1, 2, 3]])
y_t = np.array([1, 1, 1, 1])
X_val = np.array([[1, 1, 5, 1, 2, 3]])
y_val = np.array([1])
lm = logreg.LogisticRegressor(num_feats=6, learning_rate=0.4, tol=0.01, max_iter=10, batch_size=10)
lm.train_model(X_t, y_t, X_val, y_val)



def test_prediction():
    X_t = np.array([[1, 2, 3, 1, 2, 3,1], [1, 2, 2, 1, 2, 3,1], [2, 1, 4, 1, 2, 3,1], [1, 2, 5, 1, 2, 3,1]])
    x = lm.make_prediction(X_t)
    assert len(x) == len(X_t)
    assert max(x) <= 1
    assert min(x) >= 0

def test_loss_function():
    assert lm.loss_function([1,1,1], [0.1, 0.1, 0.1]) == 3
    assert lm.loss_function([1,1,1], [.99, .99, .99]) - .013 <= .0001


def test_gradient():
    X_t = np.array([[1, 2, 3, 1, 2, 3, 1], [1, 2, 2, 1, 2, 3, 1], [2, 1, 4, 1, 2, 3, 1], [1, 2, 5, 1, 2, 3, 1]])
    a = lm.calculate_gradient(y_t, X_t)
    assert len(a) == 7


def test_training():
    b = lm.get_loss()
    assert b[0] >= b[-1]
