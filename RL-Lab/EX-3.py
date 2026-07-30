# MDP for Warehouse Robot

states = ["Start", "Obstacle", "Goal"]
actions = ["Left", "Right"]

rewards = {
    ("Start", "Right"): 10,
    ("Start", "Left"): -2,
    ("Obstacle", "Right"): -5,
    ("Goal", "Right"): 100
}

print("Warehouse Robot MDP\n")

for state in states:
    print("State:", state)

    if state == "Goal":
        print("Goal Reached")
    else:
        best_action = max(actions, key=lambda a: rewards.get((state, a), -100))
        print("Best Action:", best_action)

    print()
