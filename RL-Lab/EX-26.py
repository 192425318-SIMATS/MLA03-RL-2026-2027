import numpy as np
import random

grid_size = 5
robots = 2
episodes = 100

actions = [0, 1, 2, 3]   # Up, Down, Left, Right

q_table = np.zeros((robots, grid_size, grid_size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(episodes):

    positions = [[0, 0], [4, 0]]
    goals = [[4, 4], [0, 4]]

    for step in range(50):

        for r in range(robots):

            row, col = positions[r]

            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = np.argmax(q_table[r, row, col])

            new_row, new_col = row, col

            if action == 0 and row > 0:
                new_row -= 1
            elif action == 1 and row < grid_size - 1:
                new_row += 1
            elif action == 2 and col > 0:
                new_col -= 1
            elif action == 3 and col < grid_size - 1:
                new_col += 1

            if [new_row, new_col] == positions[1-r]:
                reward = -10
            elif [new_row, new_col] == goals[r]:
                reward = 10
            else:
                reward = -1

            old = q_table[r, row, col, action]
            next_value = np.max(q_table[r, new_row, new_col])

            q_table[r, row, col, action] = old + alpha * (
                reward + gamma * next_value - old
            )

            positions[r] = [new_row, new_col]

print("MARL Training Completed")
print("Cooperative warehouse robot learning completed.")
