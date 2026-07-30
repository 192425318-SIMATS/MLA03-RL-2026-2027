import numpy as np
import random

# States
states = ["Standing", "Walking", "Running"]

# Actions
actions = ["Step Forward", "Balance", "Stop"]

# Initial Policy
policy = np.ones((3,3)) / 3

learning_rate = 0.1

for episode in range(100):

    state = random.randint(0,2)

    # Select Best Action
    action = np.argmax(policy[state])

    # Reward
    reward = random.choice([10,20,30])

    # PPO/TRPO Policy Update
    policy[state][action] += learning_rate * reward

    # Normalize Policy
    policy[state] = policy[state] / np.sum(policy[state])

print("Updated Policy:\n")
print(policy)

print("\nOptimal Actions:")
for i in range(3):
    print(states[i], "->", actions[np.argmax(policy[i])])
