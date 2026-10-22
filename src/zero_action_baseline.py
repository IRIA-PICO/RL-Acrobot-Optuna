
import numpy as np
from common import create_env

env = create_env()

# With 3 bins per action dimension,
# index 4 should correspond to [0, 0].
ZERO_ACTION = 4

print("Action 4 maps to:", env.env.action(ZERO_ACTION))

rewards = []

for episode in range(50):
    state, info = env.reset(seed=1000 + episode)

    total_reward = 0
    done = False

    while not done:
        state, reward, terminated, truncated, info = env.step(
            ZERO_ACTION
        )
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)

env.close()

print("\n=== ZERO ACTION BASELINE ===")
print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Std reward: {np.std(rewards):.2f}")
