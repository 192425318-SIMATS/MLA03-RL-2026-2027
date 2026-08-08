import numpy as np
import random

prices = [100, 105, 102, 110, 108, 115, 120, 116, 125, 130]

actions = ["Buy", "Sell", "Hold"]

policy = np.ones((len(prices), 3)) / 3

learning_rate = 0.1

for episode in range(100):

    total_profit = 0

    for i in range(len(prices) - 1):

        action = np.random.choice(3, p=policy[i])

        if action == 0:       # Buy
            reward = prices[i + 1] - prices[i]

        elif action == 1:     # Sell
            reward = prices[i] - prices[i + 1]

        else:                 # Hold
            reward = 0

        total_profit += reward

        policy[i][action] += learning_rate * reward
        policy[i] = np.maximum(policy[i], 0)
        policy[i] /= np.sum(policy[i])

print("Training Completed")
print("Total Profit:", round(total_profit, 2))

print("\nLearned Trading Policy:")

for i in range(len(prices) - 1):
    best_action = np.argmax(policy[i])
    print(prices[i], "->", actions[best_action])
