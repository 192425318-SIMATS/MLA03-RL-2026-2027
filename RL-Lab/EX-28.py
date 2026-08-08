import numpy as np

grid = 4
gamma = 0.9

V = np.zeros((grid, grid))

goal = (3, 3)
start = (0, 0)

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

for _ in range(100):

    new_V = V.copy()

    for r in range(grid):
        for c in range(grid):

            if (r, c) == goal:
                continue

            values = []

            for dr, dc in actions:

                nr = r + dr
                nc = c + dc

                if 0 <= nr < grid and 0 <= nc < grid:
                    reward = 10 if (nr, nc) == goal else -1
                    value = reward + gamma * V[nr, nc]
                    values.append(value)

            new_V[r, c] = max(values)

    V = new_V

print("Optimal State Values:")
print(np.round(V, 2))

# Find optimal path
state = start
path = [state]

while state != goal:

    r, c = state
    best_value = -float("inf")
    best_state = state

    for dr, dc in actions:

        nr = r + dr
        nc = c + dc

        if 0 <= nr < grid and 0 <= nc < grid:

            reward = 10 if (nr, nc) == goal else -1
            value = reward + gamma * V[nr, nc]

            if value > best_value:
                best_value = value
                best_state = (nr, nc)

    state = best_state
    path.append(state)

print("Optimal Path:")
print(path)
