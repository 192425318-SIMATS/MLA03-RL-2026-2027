import numpy as np

# States
states = ["Straight Road", "Traffic", "Turn"]

# Actions
actions = ["Accelerate", "Brake", "Turn"]

# Offline Reward Data
rewards = np.array([
    [30, 10, 5],      # Straight Road
    [5, 30, 10],      # Traffic
    [10, 5, 30]       # Turn
])

# Initialize Q-Table
Q = np.zeros((3, 3))

# Parameters
alpha = 0.8
gamma = 0.9

# Offline Training
for state in range(3):
    for action in range(3):

        next_state = (state + 1) % 3

        Q[state][action] = Q[state][action] + alpha * (
            rewards[state][action] +
            gamma * np.max(Q[next_state]) -
            Q[state][action]
        )

print("Learned Q-Table:")
print(Q)

print("\nOptimal Driving Policy:")
for i in range(3):
    print(states[i], "->", actions[np.argmax(Q[i])])
