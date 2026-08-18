import numpy as np

# Actions: 0 = Heating, 1 = Cooling, 2 = No Action
actions = 3
episodes = 100
learning_rate = 0.01

policy = np.ones((3, actions)) / actions

for episode in range(episodes):

    temperature = np.random.randint(15, 31)

    states = []
    selected_actions = []
    rewards = []

    for step in range(10):

        # Temperature state
        if temperature < 20:
            state = 0
        elif temperature > 25:
            state = 2
        else:
            state = 1

        # Select action
        action = np.random.choice(
            actions,
            p=policy[state]
        )

        # Environment response
        if action == 0:
            temperature += 1
            energy = 2
        elif action == 1:
            temperature -= 1
            energy = 2
        else:
            energy = 0

        # Reward
        if 20 <= temperature <= 25:
            reward = 5 - energy
        else:
            reward = -3 - energy

        states.append(state)
        selected_actions.append(action)
        rewards.append(reward)

    # Calculate returns
    returns = []
    total = 0

    for reward in reversed(rewards):
        total = reward + 0.9 * total
        returns.insert(0, total)

    # REINFORCE policy update
    for s, a, G in zip(
        states, selected_actions, returns
    ):
        policy[s][a] += learning_rate * G
        policy[s] = np.maximum(policy[s], 0)
        policy[s] /= np.sum(policy[s])

print("REINFORCE Training Completed")
print("Learned Smart Home Policy:")
print(np.round(policy, 2))
