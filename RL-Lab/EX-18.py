import numpy as np
import random

# Manufacturing Tasks
tasks = ["Welding", "Painting", "Assembly"]

# Initialize Task Values
values = np.zeros(3)

learning_rate = 0.1

# Training
for episode in range(100):

    task = random.randint(0, 2)

    # Reward
    reward = random.choice([10, 20, 30])

    # Update Task Value
    values[task] = values[task] + learning_rate * (reward - values[task])

print("Learned Task Values:")
print(values)

print("\nBest Manufacturing Task:")
print(tasks[np.argmax(values)])
