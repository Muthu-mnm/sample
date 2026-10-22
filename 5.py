"""
Implement TD learning methods (TD (0) and TD (1)) to estimate value functions.
"""

import random
states = ['Study', 'Practice', 'Revision', 'Exam']
V = {s:0 for s in states}

gamma = 0.9
alpha = 0.1

P = {
    'Study' : ('Practice', 2),
    'Practice' : ('Revision', 3),
    'Revision' : ('Exam', 5),
    'Exam' : (None, 10)
}

#TD(0)

for _ in range(100):
    state = 'Study'

    while state is not None:
        next_state, reward = P[state]

        if next_state is None:
            target = reward
        else:
            target = reward + gamma * V[next_state]
        V[state] = V[state] + alpha * (target - V[state])
        state = next_state

print("TD(0) Values: ")
for s in states:
    print(f"{s} : {V[s]:.2f}")

#TD(1)

V = {s: 0 for s in states}

for _ in range(100):
    state = "Study"
    states_seen = []
    rewards = []

    while state is not None:
        states_seen.append(state)
        next_state,reward = P[state]
        rewards.append(reward)
        state = next_state
    
    for t in range(len(states_seen)):
        G = 0
        power = 0

        for r in rewards[t:]:
            G += (gamma ** power) * r
            power += 1
        V[states_seen[t]] = V[states_seen[t]] + alpha *(G - V[states_seen[t]])
print("\nTD(1) Values: ")
for s in states:
    print(f"{s} : {V[s]:.2f}")