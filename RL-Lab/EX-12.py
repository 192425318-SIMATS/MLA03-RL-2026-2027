import numpy as np

# Actions
actions = ["Left", "Right"]

# Initial Policy Probabilities
policy = np.array([0.5, 0.5])

# Rewards
rewards = [5, 10]

# Best Action
best_action = np.argmax(rewards)

# Update Policy
policy[best_action] += 0.2
policy = policy / np.sum(policy)

print("Updated Policy:", policy)
print("Best Action:", actions[np.argmax(policy)])
