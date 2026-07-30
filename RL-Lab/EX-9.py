# Temporal Difference (TD-0) Learning

alpha = 0.5          # Learning Rate
gamma = 0.9          # Discount Factor

current_value = 10
reward = 5
next_value = 12

updated_value = current_value + alpha * (
    reward + gamma * next_value - current_value
)

print("Current State Value :", current_value)
print("Reward :", reward)
print("Next State Value :", next_value)
print("Updated State Value :", updated_value)
