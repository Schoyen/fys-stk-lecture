import math
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tf.keras.backend.set_floatx("float64")

num_points = 11

start = tf.constant(0, dtype=tf.float64)
stop = tf.constant(1, dtype=tf.float64)
stop_t = stop

X, T = tf.meshgrid(
    tf.linspace(start, stop, num_points), tf.linspace(start, stop_t, num_points)
)

x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])
coords = tf.concat([x, t], axis=1)


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(90, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)

        return self.out(x)


@tf.function
def g_analytic(x, t):
    return tf.sin(math.pi * x) * tf.cos(math.pi * t) - tf.sin(
        math.pi * x
    ) * tf.sin(math.pi * t)


@tf.function
def u(x):
    return tf.sin(math.pi * x)


@tf.function
def v(x):
    return -math.pi * tf.sin(math.pi * x)


@tf.function
def trial_solution(model, x, t):
    points = tf.concat([x, t], axis=1)

    return (1 - t ** 2) * u(x) + t * v(x) + x * (1 - x) * t ** 2 * model(points)


@tf.function
def loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch([x, t])

            trial = trial_solution(model, x, t)

        d_trial_dx = tape_2.gradient(trial, x)
        d_trial_dt = tape_2.gradient(trial, t)

    d2_trial_d2x = tape.gradient(d_trial_dx, x)
    d2_trial_d2t = tape.gradient(d_trial_dt, t)

    del tape_2
    del tape

    return tf.losses.MSE(
        tf.zeros_like(d2_trial_d2x), d2_trial_d2x - d2_trial_d2t
    )


@tf.function
def grad(model, x, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


model = DNModel()
optimizer = tf.keras.optimizers.Adam(0.01)

num_epochs = 1000

for epoch in range(num_epochs):
    cost, gradients = grad(model, x, t)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(
        f"Step: {optimizer.iterations.numpy()}, "
        + f"Loss: {tf.math.reduce_mean(cost.numpy())}"
    )


num_points = 41
X, T = tf.meshgrid(
    tf.linspace(start, stop, num_points), tf.linspace(start, stop_t, num_points)
)
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])

g = tf.reshape(g_analytic(x, t), (num_points, num_points))
g_nn = tf.reshape(trial_solution(model, x, t), (num_points, num_points))

diff = tf.abs(g - g_nn)
print(f"Max diff: {tf.reduce_max(diff)}")
print(f"Mean diff: {tf.reduce_mean(diff)}")


fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_title("Analytic")
ax.plot_surface(X, T, g)


fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_title("Neural")
ax.plot_surface(X, T, g_nn)


fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_title("Diff")
ax.plot_surface(X, T, diff)

plt.show()
