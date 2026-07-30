import numpy as np
import random

# States and Actions
states = 4
actions = 2

# Initialize Q-table
Q = np.zeros((states, actions))

alpha = 0.5      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.2    # Exploration rate

# Training
for episode in range(10):

    state = random.randint(0, states - 1)

    if random.random() < epsilon:
        action = random.randint(0, actions - 1)
    else:
        action = np.argmax(Q[state])

    reward = random.randint(0, 10)
    next_state = random.randint(0, states - 1)

    Q[state][action] = Q[state][action] + alpha * (
        reward + gamma * np.max(Q[next_state]) - Q[state][action]
    )

print("Learned Q-Table:\n")
print(Q)
