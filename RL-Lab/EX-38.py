import numpy as np
import random

states = 5
actions = 3

# Q-table
Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 100

for episode in range(episodes):

    state = random.randint(0, states - 1)

    for step in range(20):

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, actions - 1)
        else:
            action = np.argmax(Q[state])

        # Simulated healthcare transition
        next_state = random.randint(0, states - 1)

        # Reward
        if action == 0:
            reward = 5       # Good scheduling
        elif action == 1:
            reward = 3       # Resource allocation
        else:
            reward = 2       # Treatment planning

        # Penalty for poor outcome
        if random.random() < 0.2:
            reward -= 4

        # Q-learning update
        Q[state, action] += alpha * (
            reward +
            gamma * np.max(Q[next_state]) -
            Q[state, action]
        )

        state = next_state

print("Healthcare RL Training Completed")
print("Learned Healthcare Policy:")
print(np.argmax(Q, axis=1))
