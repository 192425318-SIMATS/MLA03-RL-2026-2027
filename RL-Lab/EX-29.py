import numpy as np

states = ["Low", "Medium", "High"]
actions = ["Green", "Red"]

# Waiting-time rewards
reward = {
    "Low": [5, -2],
    "Medium": [8, -5],
    "High": [12, -10]
}

gamma = 0.9

value = np.zeros(len(states))

for i in range(100):
    new_value = np.zeros(len(states))

    for s in range(len(states)):
        action_values = []

        for a in range(len(actions)):
            next_state = (s + 1) % len(states)

            v = reward[states[s]][a] + gamma * value[next_state]
            action_values.append(v)

        new_value[s] = max(action_values)

    value = new_value

print("Optimal State Values:")
for i in range(len(states)):
    print(states[i], ":", round(value[i], 2))

print("\nOptimal Traffic Light Policy:")

for s in range(len(states)):
    values = []

    for a in range(len(actions)):
        next_state = (s + 1) % len(states)
        v = reward[states[s]][a] + gamma * value[next_state]
        values.append(v)

    best_action = np.argmax(values)

    print(states[s], "->", actions[best_action])
