import numpy as np
import random

# Household Tasks
tasks = ["Cleaning", "Cooking", "Charging"]

# Task Values
values = np.zeros(3)

learning_rate = 0.1

# Training
for episode in range(100):

    task = random.randint(0, 2)

    # Reward for task completion
    reward = random.choice([10, 20, 30])

    # Update task value
    values[task] = values[task] + learning_rate * (reward - values[task])

print("Task Values:\n")
print(values)

print("\nBest Task:")
print(tasks[np.argmax(values)])
