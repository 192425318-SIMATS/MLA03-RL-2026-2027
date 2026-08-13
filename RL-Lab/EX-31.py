import numpy as np
import random

states = 9
actions = 9

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 500

for episode in range(episodes):

    state = random.randint(0, states - 1)

    if random.random() < epsilon:
        action = random.randint(0, actions - 1)
    else:
        action = np.argmax(Q[state])

    for step in range(20):

        next_state = random.randint(0, states - 1)

        reward = random.choice([1, 0, -1])

        if random.random() < epsilon:
            next_action = random.randint(0, actions - 1)
        else:
            next_action = np.argmax(Q[next_state])

        # SARSA update
        Q[state, action] = Q[state, action] + alpha * (
            reward +
            gamma * Q[next_state, next_action] -
            Q[state, action]
        )

        state = next_state
        action = next_action

print("SARSA Training Completed")

print("Learned Q-Table:")
print(np.round(Q, 2))

print("Best Actions:")
print(np.argmax(Q, axis=1))
