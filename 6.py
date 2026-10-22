""" 
    Implement the Q-learning algorithm to optimize dynamic CPU allocation under varying loads.
"""

import random
import numpy as np

states = [0, 1, 2]
actions = [25, 50, 75, 100]

q_table = np.zeros((len(states), len(actions)))

alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 500

def get_rewards(state, cpu):
    load = [30, 60, 90][state]

    if load <= 30:
        ideal = 25
    elif load <= 60:
        ideal = 50
    else: ideal = 100

    reward = -abs(cpu - ideal)

    if cpu < load:
        reward -= 20
    
    return reward

for _ in range(episodes):
    state = random.choice(states)

    for step in range(10):
        action = random.randint(0, len(actions) - 1) if random.random() < epsilon else np.argmax(q_table[state])

        cpu = actions[action]

        reward = get_rewards(state, cpu)

        next_state = random.choice(states)

        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

print("Q-Table:") 
print(q_table) 
 
print("\nOptimal CPU Allocation:") 
 
for state in states: 
    best_action = np.argmax(q_table[state]) 
    load = ["Low (30%)", "Medium (60%)", "High (90%)"][state] 
    print(load, "->", actions[best_action], "% CPU") 