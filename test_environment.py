from traffic_environment import TrafficEnvironment
import numpy as np

env = TrafficEnvironment()

state = env.reset()
print("Initial State:", state)

for step in range(10):
    action = np.random.choice([0,1])  # random action
    next_state, reward, done = env.step(action)

    print("Step:", step)
    print("State:", next_state)
    print("Reward:", reward)

    if done:
        break