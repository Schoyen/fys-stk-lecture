import numpy as np
import tensorflow as tf

import sklearn.neural_network
import sklearn.model_selection


NUM_DIGITS = 10
NUM_VALUES = int(2 ** NUM_DIGITS)


def binary_encoding(x, num_digits):
    # Make digits one-hot encoded
    return np.array([x >> i & 1 for i in range(num_digits)])


def fizz_buzz_one_hot(x):
    # One-hot encode fizz buzz solution
    if x % 15 == 0:
        # return "fizzbuzz"
        return np.array([0, 0, 0, 1])
    if x % 5 == 0:
        # return "buzz"
        return np.array([0, 0, 1, 0])
    if x % 3 == 0:
        # return "fizz"
        return np.array([0, 1, 0, 0])

    # return str(x)
    return np.array([1, 0, 0, 0])


def fizz_buzz(x, ind):
    # Output fizz buzz response from one-hot encoded solution
    return [str(x), "fizz", "buzz", "fizzbuzz"][np.argmax(ind)]


# print(binary_encoding(3, NUM_DIGITS))
# print(binary_encoding(5, NUM_DIGITS))
# print(binary_encoding(8, NUM_DIGITS))
# print(binary_encoding(7, NUM_DIGITS))

# for i in range(100):
#    print(fizz_buzz(i, fizz_buzz_one_hot(i)))

# Create training and test data
X, y = np.zeros((NUM_VALUES, NUM_DIGITS)), np.zeros((NUM_VALUES, 4))

for i in range(NUM_VALUES):
    X[i] = binary_encoding(i, NUM_DIGITS)
    y[i] = fizz_buzz_one_hot(i)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3
)

# Train (sklearn) neural network classifier

# Output accuracy

# Output response of the first 100 numbers


# Train (Tensorflow/Keras) neural network classifier
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30),
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
)


# Compile model
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Fit to training data
model.fit(
    X_train,
    y_train,
    epochs=400,
    batch_size=32,
    validation_data=(X_test, y_test),
)

# Output response of the first 100 numbers
for i in range(100):
    pred = model.predict(binary_encoding(i, NUM_DIGITS).reshape(1, -1))

    print(
        f"{fizz_buzz(i, fizz_buzz_one_hot(i)):<10} "
        + f"|| {fizz_buzz(i, pred)}"
    )
