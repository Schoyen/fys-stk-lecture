import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics

np.random.seed(2019)


def franke(x, y):
    term = 3 / 4 * np.exp(-(9 * x - 2) ** 2 / 4 - (9 * y - 2) ** 2 / 4)
    term += 3 / 4 * np.exp(-(9 * x + 1) ** 2 / 49 - (9 * y + 1) / 10)
    term += 1 / 2 * np.exp(-(9 * x - 7) ** 2 / 4 - (9 * y - 3) ** 2 / 4)
    term -= 1 / 5 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)

    return term


L = 41

X, Y = np.meshgrid(np.linspace(0, 1, L), np.linspace(0, 1, L))
Z = franke(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z)
ax.set_title("Franke's function")

plt.show()

X_d = np.c_[X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]]
y_d = Z.ravel()[:, np.newaxis]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X_d, y_d, test_size=0.2
)


# Implement nueral network

# See some statistics

# Plot surface fit
