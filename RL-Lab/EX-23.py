import numpy as np
import random

lanes = 3
episodes = 100

actions = ["Stay", "Left", "Right"]

q = np.zeros((lanes, 3))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(episodes):

    lane = 1
    speed = random.randint(40, 80)

    for step in range(50):

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(q[lane])

        if action == 1 and lane > 0:
            lane -= 1

        elif action == 2 and lane < lanes - 1:
            lane += 1

        traffic_speed = random.randint(30, 80)

        if speed > traffic_speed:
            reward = 5
        else:
            reward = -2

        old_value = q[lane][action]
        next_value = np.max(q[lane])

        q[lane][action] = old_value + alpha * (
            reward + gamma * next_value - old_value
        )

print("Training completed")
print("Learned Lane-Changing Policy:")

for lane in range(lanes):
    best_action = np.argmax(q[lane])
    print("Lane", lane + 1, "->", actions[best_action])
