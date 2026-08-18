import numpy as np
import random

# States and subtasks
states = 5

subtasks = {
    "Collect": 0,
    "Move": 1,
    "Deliver": 2
}

Q = np.zeros((len(subtasks), states, 2))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 100

for episode in range(episodes):

    state = 0

    for task_name, task in subtasks.items():

        for step in range(10):

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = np.argmax(Q[task, state])

            # Simulated transition
            next_state = min(
                states - 1,
                state + random.choice([0, 1])
            )

            # Reward for completing a subtask
            if next_state == states - 1:
                reward = 10
            else:
                reward = -1

            # MAXQ subtask update
            Q[task, state, action] += alpha * (
                reward +
                gamma * np.max(Q[task, next_state]) -
                Q[task, state, action]
            )

            state = next_state

print("MAXQ Training Completed")
print("\nLearned Subtask Policies:")

for task_name, task in subtasks.items():
    print(
        task_name,
        "->",
        np.argmax(Q[task], axis=1)
    )

print("\nHierarchical cooperative task completed.")
