import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math


def step_activation(x: npt.NDArray) -> npt.NDArray:
    """Step (Heaviside) activation: 1 if x >= 0 else 0.

    `x >= 0` produces a boolean array; `.astype(x.dtype)` casts True/False to
    1/0 while keeping the input's dtype (so float32 in => float32 out, which
    keeps downstream matmuls happy).
    """
    return (x >= 0).astype(x.dtype)


def sigmoid(inputs: npt.NDArray) -> npt.NDArray:
    """Sigmoid activation: squashes any real number into (0, 1)."""
    return 1 / (1 + np.exp(-inputs))


def relu(inputs: npt.NDArray) -> npt.NDArray:
    """ReLU activation: max(0, x), applied elementwise."""
    return np.maximum(0, inputs)


def softmax(inputs: npt.NDArray) -> npt.NDArray:
    """Softmax activation: turn a row of logits into a probability distribution.

    Expects a 2D input shaped (batch_size, num_classes); the normalization is
    done per-row via `axis=1`.
    """

    # the - np.max(...) solution seems to be to avoiod dead neurons
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    return probabilities
