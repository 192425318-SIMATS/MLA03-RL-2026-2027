# Experiment 11
# Compare DQN with Traditional Q-Learning

# Sample rewards obtained after training
q_learning_rewards = [12, 18, 15, 20, 17]
dqn_rewards = [20, 25, 23, 28, 26]

# Average rewards
q_avg = sum(q_learning_rewards) / len(q_learning_rewards)
dqn_avg = sum(dqn_rewards) / len(dqn_rewards)

print("Average Reward (Q-Learning):", q_avg)
print("Average Reward (DQN):", dqn_avg)

if dqn_avg > q_avg:
    print("DQN performs better than Traditional Q-Learning.")
else:
    print("Q-Learning performs better.")
