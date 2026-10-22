# Develop a simple grid-world environment where the agent learns to navigate using predefined states, actions, and rewards. 

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation


## Environment

SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLE = (2, 2)
ACTIONS = [(-1, 0), (1, 0), (0,-1), (0, 1)]

# step function

def step(state, action):
    dx, dy = ACTIONS[action]

    x = min(max(state[0] + dx, 0), SIZE - 1)
    y = min(max(state[1] + dy, 0), SIZE - 1)

    next_state = (x, y)

    if next_state == GOAL:
        return next_state, 100, True
    elif next_state == OBSTACLE:
        return next_state, -100, True
    else:
        return next_state, -1, False

## Q- Learning

q_table = np.zeros((SIZE, SIZE, 4))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 500


for _ in range(episodes):
    state = START
    done = False

    while not done:
        x, y  = state

        action = random.randint(0, 3) if random.random() < epsilon else np.argmax(q_table[x,y])

        next_state, reward, done = step(state, action)

        nx, ny = next_state
        q_table[x, y, action] += alpha * (reward + gamma * np.max(q_table[nx,ny]) - q_table[x, y, action])

        state = next_state
print("Training Completed")

## Extract learned path

state = START
path = [state]

for _ in range(30):
    x, y = state
    action = np.argmax(q_table[x,y])

    state, reward, done = step(state, action)
    path.append(state)

    if done:
        break

    

print("Path: ", path)
print("Final State: ", state)

## Visualization

fig, ax = plt.subplots(figsize = (5,5))

def animate(i):
    ax.clear()

    ax.set_xlim(0, SIZE)
    ax.set_ylim(0, SIZE)

    ax.set_xticks(range(SIZE + 1))
    ax.set_yticks(range(SIZE + 1))

    ax.grid(True)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.add_patch(plt.Rectangle((GOAL[1], GOAL[0]), 1, 1, color = "gold"))
    ax.add_patch(plt.Rectangle((OBSTACLE[1], OBSTACLE[0]), 1, 1, color = "red"))
    ax.add_patch(plt.Rectangle((START[1], START[0]), 1, 1, color = "lightgreen"))

    #trail
    trail = path[:i+1]
    tx = [c + 0.5 for r, c in trail]
    ty = [r + 0.5 for r, c in trail]
    ax.plot(tx, ty, color = "blue", linewidth = 2, alpha = 0.6)
    
    #agent
    r, c = path[i]
    ax.add_patch(plt.Circle((c + 0.5, r + 0.5), 0.3, color = "blue"))

    ax.set_title("Learned Path")

ani = animation.FuncAnimation(fig, animate, frames = len(path), interval = 500, repeat = False)
plt.show()