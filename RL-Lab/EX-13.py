import numpy as np

# Actions
actions = ["Left", "Right"]

# Initial Policy
policy = np.array([0.5, 0.5])

# Rewards
rewards = [4, 8]

# Select Best Action
best_action = np.argmax(rewards)

# REINFORCE Policy Update
learning_rate = 0.1
policy[best_action] += learning_rate * rewards[best_action]

# Normalize Policy
policy = policy / np.sum(policy)

print("Updated Policy:", policy)
print("Selected Action:", actions[np.argmax(policy)])
