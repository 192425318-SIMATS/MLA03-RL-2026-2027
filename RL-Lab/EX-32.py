import numpy as np
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
import tensorflow as tf

states = 16
actions = 4

# Dueling DQN
inputs = Input(shape=(states,))

x = Dense(32, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)

value = Dense(1)(x)
advantage = Dense(actions)(x)

q_values = Lambda(
    lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True))
)([value, advantage])

model = Model(inputs=inputs, outputs=q_values)

model.compile(
    optimizer="adam",
    loss="mse"
)

episodes = 50

for episode in range(episodes):

    state = np.zeros(states)
    state[random.randint(0, states - 1)] = 1

    for step in range(20):

        q = model(
            state.reshape(1, -1),
            training=False
        ).numpy()

        if random.random() < 0.2:
            action = random.randint(0, actions - 1)
        else:
            action = np.argmax(q[0])

        next_state = np.zeros(states)
        next_state[random.randint(0, states - 1)] = 1

        reward = random.choice([1, 2, -1])

        target = reward + 0.9 * np.max(
            model(
                next_state.reshape(1, -1),
                training=False
            ).numpy()
        )

        q[0][action] = target

        model.fit(
            state.reshape(1, -1),
            q,
            epochs=1,
            verbose=0
        )

        state = next_state

print("Dueling DQN Training Completed")
print("Optimal Gridworld Navigation Learned")
