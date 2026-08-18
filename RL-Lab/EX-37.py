import numpy as np
import random

# Grid environment
grid_size = 5

# Possible actions
actions = ["Up", "Down", "Left", "Right"]

# Actual robot position
robot = [0, 0]

# Goal position
goal = [4, 4]

# Belief state
belief = np.ones((grid_size, grid_size))
belief = belief / np.sum(belief)

for step in range(30):

    # Select an action
    action = random.choice(actions)

    # Move robot
    if action == "Up" and robot[0] > 0:
        robot[0] -= 1

    elif action == "Down" and robot[0] < grid_size - 1:
        robot[0] += 1

    elif action == "Left" and robot[1] > 0:
        robot[1] -= 1

    elif action == "Right" and robot[1] < grid_size - 1:
        robot[1] += 1

    # Simulated noisy observation
    if random.random() < 0.8:
        observation = tuple(robot)
    else:
        observation = (
            random.randint(0, grid_size - 1),
            random.randint(0, grid_size - 1)
        )

    # Update belief based on observation
    belief *= 0.1
    belief[observation] = 1.0
    belief /= np.sum(belief)

    # Check goal
    if robot == goal:
        print("Robot reached the goal.")
        break

print("POMDP Navigation Completed")
print("Final Robot Position:", robot)
print("Most Likely Position:",
      np.unravel_index(np.argmax(belief), belief.shape))
