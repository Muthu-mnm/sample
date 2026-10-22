"""
    Use Monte Carlo methods to estimate the value function and develop a policy for episodic tasks.
"""

import random
states = ['Poor', 'Average','Good']
action = ['Study', 'Rest']
gamma = 0.9

P = {
    'Poor' : {
        'Study' : ('Average', 5),
        'Rest' : ('Poor', -1)
    },
    'Average' : {
        'Study' : ('Good', 8),
        'Rest' : ('Poor', -2)
    },
    'Good' : {
        'Study' : ('Good', 2),
        'Rest' : ('Good', 0)
    }
}

episodes = 500
V = {s : 0 for s in states}
returns = {s : [] for s in states}

for _ in range(episodes):
    s = 'Poor'
    episode_data = []

    for i in range(10):
        a = random.choice(action)
        next_state, reward = P[s][a]
        episode_data.append((s,reward))
        s = next_state
    
    G = 0
    visited = set()

    for state, reward in reversed(episode_data):
        G = reward + gamma * G

        if state not in visited:
            returns[state].append(G)
            V[state] = sum(returns[state]) / len(returns[state])
            visited.add(state)
policy = {}

for s in states:
    action_values = []
    for a in action:
        next_state, reward = P[s][a]
        action_values.append(reward + gamma * V[s])
    policy[s] = max(action_values)

print("\nEstimated Value Function:")
for state in states:
    print(f"{state}: {V[state]:.2f}")

print("\nPolicy:")
for state in states:
    print(f"{state} --> {policy[state]:.4f}")
