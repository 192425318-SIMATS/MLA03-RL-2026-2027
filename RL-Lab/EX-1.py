# MDP for Simplified Chess Game

states = ["Start", "Middle", "End"]
actions = ["Attack", "Defend"]

rewards = {
    ("Start", "Attack"): 5,
    ("Start", "Defend"): 2,
    ("Middle", "Attack"): 10,
    ("Middle", "Defend"): 6,
    ("End", "Attack"): 20,
    ("End", "Defend"): 8
}

for state in states:
    best_action = max(actions, key=lambda action: rewards[(state, action)])
    print("State:", state)
    print("Best Action:", best_action)
    print()

    
