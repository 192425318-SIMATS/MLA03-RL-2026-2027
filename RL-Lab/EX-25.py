import numpy as np

ads = 3
rounds = 1000

true_ctr = [0.05, 0.08, 0.12]

# Epsilon-Greedy
eps = 0.1
counts = np.zeros(ads)
rewards = np.zeros(ads)

for i in range(rounds):
    if np.random.rand() < eps:
        ad = np.random.randint(ads)
    else:
        ad = np.argmax(rewards / (counts + 1e-9))

    click = np.random.rand() < true_ctr[ad]
    counts[ad] += 1
    rewards[ad] += click

epsilon_ctr = rewards.sum() / rounds


# UCB
counts = np.zeros(ads)
rewards = np.zeros(ads)

for i in range(rounds):
    if i < ads:
        ad = i
    else:
        mean = rewards / (counts + 1e-9)
        ucb = mean + np.sqrt(2 * np.log(i + 1) / (counts + 1e-9))
        ad = np.argmax(ucb)

    click = np.random.rand() < true_ctr[ad]
    counts[ad] += 1
    rewards[ad] += click

ucb_ctr = rewards.sum() / rounds


# Thompson Sampling
success = np.ones(ads)
failure = np.ones(ads)

for i in range(rounds):
    samples = [
        np.random.beta(success[j], failure[j])
        for j in range(ads)
    ]

    ad = np.argmax(samples)

    click = np.random.rand() < true_ctr[ad]

    if click:
        success[ad] += 1
    else:
        failure[ad] += 1

thompson_ctr = sum(success - 1) / rounds

print("Epsilon-Greedy CTR:", round(epsilon_ctr, 3))
print("UCB CTR:", round(ucb_ctr, 3))
print("Thompson Sampling CTR:", round(thompson_ctr, 3))
