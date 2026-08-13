import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

state_size = 4
action_size = 2

# Actor Network
actor = Sequential([
    Dense(32, activation="relu", input_shape=(state_size,)),
    Dense(32, activation="relu"),
    Dense(action_size, activation="tanh")
])

# Critic Network
critic = Sequential([
    Dense(32, activation="relu", input_shape=(state_size + action_size,)),
    Dense(32, activation="relu"),
    Dense(1)
])

actor.compile(optimizer=Adam(0.001), loss="mse")
critic.compile(optimizer=Adam(0.001), loss="mse")

episodes = 50

for episode in range(episodes):

    state = np.random.rand(state_size)
    total_reward = 0

    for step in range(20):

        state_input = state.reshape(1, -1)

        # Actor selects action
        action = actor(state_input, training=False).numpy()[0]

        # Simulated game environment
        next_state = np.random.rand(state_size)

        resource_reward = np.random.choice([1, 2, 3])
        opponent_reward = np.random.choice([0, 2, 4])

        reward = resource_reward + opponent_reward

        # Critic training
        critic_input = np.concatenate(
            [state_input, action.reshape(1, -1)],
            axis=1
        )

        critic.fit(
            critic_input,
            np.array([[reward]]),
            epochs=1,
            verbose=0
        )

        state = next_state
        total_reward += reward

    if episode % 10 == 0:
        print("Episode:", episode, "Reward:", total_reward)

print("DDPG Training Completed")
print("Strategy Game Agent Training Completed")
