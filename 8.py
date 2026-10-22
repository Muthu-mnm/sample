""" 
  IMPLEMENT A DEEP Q-NETWORK (DQN) USING NEURAL NETWORKS TO APPROXIMATE Q VALUES. 
"""
import torch
import torch.nn as nn
import random

# Neural network
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
gamma = 0.9

# Training
for episode in range(1000):
    state = torch.tensor([random.random(), random.random()])
    next_state = torch.tensor([random.random(), random.random()])
    action = random.randint(0, 1)
    reward = 1.0

    q_value = model(state)[action]
    target = reward + gamma * model(next_state).max().detach()

    loss = (q_value - target) ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Q-value: {q_value.item():.4f}, Loss: {loss.item():.4f}")

# Final Q-values
state = torch.tensor([0.5, 0.5])
q_values = model(state).detach().numpy()

print("\nFinal Q-values:", q_values)
print("Best action:", q_values.argmax())
print("Training completed!")