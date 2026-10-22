import torch
import torch.nn as nn
import random
from collections import deque

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
memory = deque(maxlen=100)
gamma = 0.9

# Store experiences
for _ in range(50):
    memory.append((
        [random.random(), random.random()],
        random.randint(0, 1), 1.0,
        [random.random(), random.random()], False
    ))

# Experience replay training
for episode in range(100):
    batch = random.sample(memory, 8)

    for s, a, r, ns, done in batch:
        s, ns = torch.tensor(s), torch.tensor(ns)
        q = model(s)[a]
        target = r + gamma * model(ns).max().detach()

        loss = (q - target) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (episode + 1) % 20 == 0:
        print("Episode:", episode + 1, "Loss:", round(loss.item(), 4))

print("Q-values:", model(torch.tensor([0.5, 0.5])).detach().numpy())