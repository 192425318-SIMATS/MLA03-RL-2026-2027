import numpy as np
import random

# States (Elevator Floors)
states = ["Floor 1", "Floor 2", "Floor 3"]

# Actions
actions = ["Move Up", "Move Down", "Stay"]

# Actor Policy
actor = np.ones((3,3)) / 3

# Critic Values
critic = np.zeros(3)

alpha = 0.1
gamma = 0.9

for episode in range(100):

    state = random.randint(0,2)

    # Actor chooses action
    action = np.argmax(actor[state])

    # Reward
    reward = random.choice([10,20,30])

    next_state = random.randint(0,2)

    # TD Error
    td_error = reward + gamma * critic[next_state] - critic[state]

    # Critic Update
    critic[state] += alpha * td_error

    # Actor Update
    actor[state][action] += alpha * td_error

print("Actor Policy:")
print(actor)

print("\nCritic Values:")
print(critic)

print("\nBest Elevator Actions:")
for i in range(3):
    print(states[i], "->", actions[np.argmax(actor[i])])
