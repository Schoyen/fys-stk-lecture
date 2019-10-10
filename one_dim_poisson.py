import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx("float64")


def g_analytic(x):
    return x * (1 - x) * tf.exp(x)


num_points = 11

start = tf.constant(0, dtype=tf.float64)
stop = tf.constant(1, dtype=tf.float64)
x = tf.reshape(tf.linspace(start, stop, num_points), (-1, 1))


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(20, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        return self.out(x)


@tf.function
def rhs(x):
    return (3 * x + x ** 2) * tf.exp(x)


@tf.function
def trial_sol(model, x):
    return x * (1 - x) * model(x)


@tf.function
def loss(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        with tf.GradientTape() as tape_2:
            tape_2.watch(x)

            trial = trial_sol(model, x)

        d_trial = tape_2.gradient(trial, x)
    d2_trial = tape.gradient(d_trial, x)

    return tf.losses.MSE(tf.zeros_like(d2_trial), -d2_trial - rhs(x))


@tf.function
def grad(model, x):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


model = DNModel()
optimizer = tf.keras.optimizers.Adam(0.01)


num_epochs = 2000
for epoch in range(num_epochs):
    cost, gradients = grad(model, x)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(
        f"Step: {optimizer.iterations.numpy()}, "
        + f"Loss: {tf.reduce_mean(cost.numpy())}"
    )


x = tf.reshape(tf.linspace(start, stop, 1001), (-1, 1))

plt.plot(x, trial_sol(model, x), label="Neural")
plt.plot(x, g_analytic(x), label="Analytic")

plt.legend(loc="best")
plt.show()
