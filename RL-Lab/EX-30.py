import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

states = 4
actions = 3

model = Sequential([
    Dense(24, input_shape=(states,), activation="relu"),
    Dense(24, activation="relu"),
    Dense(actions, activation="linear")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

epsilon = 1.0
gamma = 0.95

for episode in range(100):

    state = np.random.rand(states)
    total_reward = 0

    for step in range(50):

        if random.random() < epsilon:
            action = random.randrange(actions)
        else:
            q_values = model.predict(
                state.reshape(1, -1), verbose=0
            )
            action = np.argmax(q_values[0])

        # Simulated highway reward
        reward = random.choice([1, 2, 3, -1])

        next_state = np.random.rand(states)

        target = reward + gamma * np.max(
            model.predict(next_state.reshape(1, -1), verbose=0)
        )

        q_values = model.predict(
            state.reshape(1, -1), verbose=0
        )

        q_values[0][action] = target

        model.fit(
            state.reshape(1, -1),
            q_values,
            epochs=1,
            verbose=0
        )

        state = next_state
        total_reward += reward

    epsilon = max(0.05, epsilon * 0.95)

print("DQN Training Completed")
print("Autonomous vehicle learning completed.")
