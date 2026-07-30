import random

# Actions (Arms)
actions = ["A", "B", "C"]

epsilon = 0.2

print("Epsilon-Greedy Action Selection\n")

for i in range(10):

    if random.random() < epsilon:
        print("Iteration", i+1, ": Explore ->", random.choice(actions))
    else:
        print("Iteration", i+1, ": Exploit -> A")
