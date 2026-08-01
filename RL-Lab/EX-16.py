import numpy as np
import random

# States (Vehicle Position)
states = ["Left Lane", "Center Lane", "Right Lane"]

# Actions
actions = ["Steer Left", "Stay", "Steer Right"]

# Initialize Policy
policy = np.ones((3,3)) / 3

learning_rate = 0.1

for episode in range(100):

    state = random.randint(0,2)

    # Select Action
    action = np.argmax(policy[state])

    # Reward
    reward = random.choice([10, 20, 30])

    # Policy Gradient Update
    policy[state][action] += learning_rate * reward

    # Normalize Policy
    policy[state] = policy[state] / np.sum(policy[state])

print("Updated Policy:\n")
print(policy)

print("\nOptimal Lane-Keeping Actions:")

for i in range(3):
    print(states[i], "->", actions[np.argmax(policy[i])])
