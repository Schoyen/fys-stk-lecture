import numpy as np
import tensorflow as tf

import sklearn.neural_network
import sklearn.model_selection


NUM_DIGITS = 10
NUM_VALUES = int(2 ** NUM_DIGITS)


def binary_encoding(x: int, num_digits: int) -> np.array:
    return np.array([x >> i & 1 for i in range(num_digits)])


def fizz_buzz_one_hot(x: int) -> np.array:
    if x % 15 == 0:
        return np.array([0, 0, 0, 1])
    if x % 5 == 0:
        return np.array([0, 0, 1, 0])
    if x % 3 == 0:
        return np.array([0, 1, 0, 0])

    return np.array([1, 0, 0, 0])


def fizz_buzz(x: int, ind: np.array) -> str:
    return [str(x), "fizz", "buzz", "fizzbuzz"][np.argmax(ind)]


X, y = np.zeros((NUM_VALUES, NUM_DIGITS)), np.zeros((NUM_VALUES, 4))

for i in range(NUM_VALUES):
    X[i] = binary_encoding(i, NUM_DIGITS)
    y[i] = fizz_buzz_one_hot(i)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3
)

clf = sklearn.neural_network.MLPClassifier(
    (100,), tol=1e-4, learning_rate="adaptive", max_iter=10000, verbose=True
).fit(X_train, y_train)

print(f"Accuracy sklearn: {clf.score(X_test, y_test)}")

for i in range(100):
    pred = clf.predict(binary_encoding(i, NUM_DIGITS).reshape(1, -1))
    print(
        f"{fizz_buzz(i, fizz_buzz_one_hot(i)):<10} "
        + f"|| {fizz_buzz(i, pred)}"
    )

input()

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30),
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=400,
    batch_size=32,
    validation_data=(X_test, y_test),
)

for i in range(100):
    pred = model.predict(binary_encoding(i, NUM_DIGITS).reshape(1, -1))
    print(
        f"{fizz_buzz(i, fizz_buzz_one_hot(i)):<10} "
        + f"|| {fizz_buzz(i, pred)}"
    )
