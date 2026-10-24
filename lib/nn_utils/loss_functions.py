import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def cat_cross_entropy(output: npt.NDArray, target_output: npt.NDArray | list):
    """target_output: one hot encoding of the correct class"""

    for i, val in enumerate(target_output):
        if val == 1:
            return -math.log(output[i])

    raise ValueError("No 1 found in target_output")


x = np.linspace(0, 1, 100)
mask = x != 0
xs = x[mask]
ys = -np.log(xs)


plt.plot(xs, ys)
plt.show()
