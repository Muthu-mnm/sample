import numpy as np 
import random 
# Grid size 
size = 5 
# Start, goal and obstacle 
start = (0, 0) 
goal = (4, 4) 
obstacle = (2, 2) 
# Actions: Up, Down, Left, Right 
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
Q = np.zeros((size, size, 4)) 
alpha = 0.1 
gamma = 0.9 
epsilon = 0.2 
# Training 
for episode in range(1000): 
    state = start 
    for step in range(50): 
        # Choose action 
        if random.random() < epsilon: 
            action = random.randint(0, 3) 
        else: 
            action = np.argmax(Q[state[0], state[1]]) 
        # Move robot 
        r, c = state 
        dr, dc = actions[action] 
        nr = max(0, min(size - 1, r + dr)) 
        nc = max(0, min(size - 1, c + dc)) 
        next_state = (nr, nc) 
        # Calculate reward 
        if next_state == goal: 
            reward = 100 
        elif next_state == obstacle: 
            reward = -100 
        else: 
            reward = -1 
        # Update Q-value 
        Q[r, c, action] += alpha * ( 
            reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action] 
        ) 
        state = next_state 
        if state == goal or state == obstacle: 
            break 
# Find learned path 
state = start 
path = [state] 
for step in range(30): 
    action = np.argmax(Q[state[0], state[1]]) 
    r, c = state 
    dr, dc = actions[action] 
    state = ( 
        max(0, min(size - 1, r + dr)), 
        max(0, min(size - 1, c + dc)) 
    ) 
    path.append(state) 
    if state == goal: 
        break 
print("Robot Path:") 
print(path)