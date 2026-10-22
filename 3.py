"""
Implement Value Iteration algorithm to find the optimal policy for an MDP.
"""

import numpy as np

states = ['Low', 'Medium', 'High']
action = ['Keep Green', 'Switch Signal']
gamma = 0.9
theta = 0.001

P = {
    'Low' : {
        'Keep Green' : ('Low', 8),
        'Switch Signal' : ('Medium', 2)
    },
    'Medium' : {
        'Keep Green' : ('Medium', 3),
        'Switch Signal' : ('Low', 7)
    },
    'High' : {
        'Keep Green' : ('High', -10),
        'Switch Signal' : ('Medium', 12)
    }
}

V = {s : 0 for s in states}

iteration = 0

while True:
    iteration+=1
    new_V = V.copy()
    delta = 0
    print("\nIteration:",iteration)

    for s in states:
        action_values = []
        for a in action:
            next_state, reward = P[s][a]
            value = reward + gamma * V[next_state]
            action_values.append(value)
            print(f"{s:6} | {a:13} -> Reward = {reward:3} Value = {value:.2f}")
        new_V[s] = max(action_values)
        delta = max(delta, abs(new_V[s] - V[s]))
    V = new_V
    print("\nUpdated State Values")
    for state in states:
        print(f"{state:6} : {V[state]:.2f}")
    
    if delta < theta:
        break
policy = {}

for s in states:
    best_value = float('-inf')
    best_action = None
    for a in action:
        next_state, reward = P[s][a]
        value = reward + gamma * V[s]

        if value > best_value:
            best_value = value
            best_action = a
    policy[s] = best_action

print("\nFinal State Values")
for state in states:
    print(f"{state:6} : {V[state]:.2f}")

print("\nOptimal Policy")

for state in states:
    print(f"{state:6} --> {policy[state]}")
