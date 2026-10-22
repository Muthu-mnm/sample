""" 
Implement Bellman Expectation Equations and solve the value function for a given MDP using Policy Iteration.
"""
import numpy as np

## Environment
states = [0, 1, 2]
actions = ['L', 'R']
gamma = 0.9

P = {
    0 : {"L" : (0, 0), "R" : (1, 5)},
    1 : {"L" : (0, 0), "R" : (2, 5)},
    2 : {"L" : (2, 0), "R" : (2, 0)}
}

policy = {
    0 : "L",
    1 : "L",
    2 : "L"
}

## policy evaluation

def policy_evaluation(policy, theta = 1e-6):
    V = np.zeros(len(states))

    while True:
        delta = 0

        for s in states:
            action = policy[s]
            next_state, reward = P[s][action]

            new_value = reward + gamma * V[next_state]
            delta = max(delta, np.abs(new_value - V[s]))
            V[s] = new_value
        if delta < theta:
            break
    return V

## policy improvement

def policy_improvement(V):
    stable = True
    
    for s in states:
        old_action = policy[s]
        action_values = {}

        for a in actions:
            next_state, reward = P[s][a]

            action_values[a] = reward + gamma * V[next_state]
        best_action = max(action_values, key = action_values.get)
        policy[s] = best_action

        if best_action != old_action:
            stable = False
    return stable

## policy iteration

iteration = 1

while True:
    print(f"\nIteration : {iteration}")

    V = policy_evaluation(policy)

    print("\nState Values:")
    for s in states:
        print(f"V(S{s}) : {V[s] : .2f}")
    
    stable = policy_improvement(V)
    for s in states:
        print(f"State {s} -> Policy {policy[s]}")
    if stable:
        break
    else : iteration += 1

print("\n===================================")
print("Optimal Policy Found")
print("===================================")

print("\nFinal Value Function")

# Display final value function.
for s in states:
    print(f"V(S{s}) = {V[s]:.2f}")

print("\nOptimal Policy")

# Display optimal action for every state.
for s in states:
    print(f"State {s} -> {policy[s]}")
