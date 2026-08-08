import numpy as np
import random

grid_size = 4
actions = [0, 1, 2, 3]  # Up, Down, Left, Right

q_table = np.zeros((grid_size, grid_size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

goal = (3, 3)
penalty = (1, 1)

for episode in range(100):
    state = (0, 0)

    for step in range(50):

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        r, c = state

        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and r < grid_size - 1:
            r += 1
        elif action == 2 and c > 0:
            c -= 1
        elif action == 3 and c < grid_size - 1:
            c += 1

        next_state = (r, c)

        if next_state == goal:
            reward = 10
            done = True
        elif next_state == penalty:
            reward = -10
            done = True
        else:
            reward = -1
            done = False

        old_value = q_table[state[0], state[1], action]
        next_value = np.max(q_table[next_state[0], next_state[1]])

        q_table[state[0], state[1], action] = old_value + alpha * (
            reward + gamma * next_value - old_value
        )

        state = next_state

        if done:
            break

print("Training completed!")
print("Q-Table:")
print(np.round(q_table, 2))

print("\nBest action at starting position:",
      np.argmax(q_table[0, 0]))
