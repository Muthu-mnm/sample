""" 
    IMPLEMENT LINEAR SQUARES TD (LSTD) FOR FUNCTION APPROXIMATION IN REINFORCEMENT LEARNING TASKS.
"""
import numpy as np 
 
# States: Low, Medium, High energy usage 
states = ["Low", "Medium", "High"] 
 
# Feature vectors 
features = { 
    "Low": np.array([1, 0, 0]), 
    "Medium": np.array([1, 1, 0]), 
    "High": np.array([1, 1, 1]) 
} 
 
gamma = 0.9 
 
# Initialize LSTD matrices 
n = 3 
A = np.zeros((n, n)) 
b = np.zeros(n) 
 
# Experience: current state, reward, next state 
experiences = [ 
    ("Low", 5, "Medium"), 
    ("Medium", 3, "High"), 
    ("High", -2, "High"), 
    ("Low", 5, "Medium"), 
    ("Medium", 3, "High"), 
    ("High", -2, "High") 
] 
 
# LSTD learning 
for state, reward, next_state in experiences: 
    phi = features[state] 
    next_phi = features[next_state] 
 
    A += np.outer(phi, phi - gamma * next_phi) 
    b += phi * reward 
 
# Calculate weights 
w = np.linalg.solve(A + 0.001 * np.eye(n), b) 
 
# Estimate value function 
print("Estimated Value Function:") 
 
for state in states: 
    value = np.dot(features[state], w) 
    print(state, ":", round(value, 2)) 