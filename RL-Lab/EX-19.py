import numpy as np
import random

# Warehouse Robots
agents = ["Robot A", "Robot B", "Robot C"]

# Initialize Agent Values
values = np.zeros(3)

learning_rate = 0.1

# Training
for episode in range(100):

    agent = random.randint(0, 2)

    # Reward
    reward = random.choice([10, 20, 30])

    # Update Agent Value
    values[agent] = values[agent] + learning_rate * (reward - values[agent])

print("Agent Values:")
print(values)

print("\nBest Performing Robot:")
print(agents[np.argmax(values)])
