import numpy as np

prices = [80, 100, 120]
demand = [0.9, 0.7, 0.5]

rounds = 1000

# Epsilon-Greedy
epsilon = 0.1
counts = np.zeros(3)
rewards = np.zeros(3)

for i in range(rounds):
    if np.random.rand() < epsilon:
        choice = np.random.randint(3)
    else:
        choice = np.argmax(rewards / (counts + 1e-9))

    sale = np.random.rand() < demand[choice]
    revenue = prices[choice] if sale else 0

    counts[choice] += 1
    rewards[choice] += revenue

epsilon_revenue = rewards.sum()


# UCB
counts = np.zeros(3)
rewards = np.zeros(3)

for i in range(rounds):
    if i < 3:
        choice = i
    else:
        mean = rewards / (counts + 1e-9)
        ucb = mean + np.sqrt(2 * np.log(i + 1) /
                              (counts + 1e-9))
        choice = np.argmax(ucb)

    sale = np.random.rand() < demand[choice]
    revenue = prices[choice] if sale else 0

    counts[choice] += 1
    rewards[choice] += revenue

ucb_revenue = rewards.sum()


# Thompson Sampling
success = np.ones(3)
failure = np.ones(3)
thompson_revenue = 0

for i in range(rounds):

    samples = [
        np.random.beta(success[j], failure[j])
        for j in range(3)
    ]

    choice = np.argmax(samples)

    sale = np.random.rand() < demand[choice]
    revenue = prices[choice] if sale else 0

    thompson_revenue += revenue

    if sale:
        success[choice] += 1
    else:
        failure[choice] += 1


print("Epsilon-Greedy Revenue:", round(epsilon_revenue, 2))
print("UCB Revenue:", round(ucb_revenue, 2))
print("Thompson Sampling Revenue:", round(thompson_revenue, 2))
