import numpy as np
import random

# States (Patient Condition)
states = ["Stable", "Moderate", "Critical"]

# Actions (Treatment)
actions = ["Medication", "Observation", "Emergency Care"]

# Reward Matrix
rewards = np.array([
    [20, 10, -10],    # Stable
    [10, 20, 15],     # Moderate
    [-20, 10, 30]     # Critical
])

# Initialize Q-Table
Q = np.zeros((3, 3))

# Parameters
alpha = 0.8
gamma = 0.9
epsilon = 0.2

# Training
for episode in range(100):

    state = random.randint(0, 2)

    # ε-Greedy Action Selection
    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        action = np.argmax(Q[state])

    next_state = random.randint(0, 2)

    # Q-Learning Update
    Q[state][action] = Q[state][action] + alpha * (
        rewards[state][action] +
        gamma * np.max(Q[next_state]) -
        Q[state][action]
    )

print("Learned Q-Table:")
print(Q)

print("\nOptimal Treatment Policy:")

for i in range(len(states)):
    print(states[i], "->", actions[np.argmax(Q[i])])
