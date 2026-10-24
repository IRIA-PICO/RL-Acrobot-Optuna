
from collections import Counter
from pathlib import Path
import numpy as np
from common import create_env

# Load the trained policy
qtable = np.load(
    "results/models/expected_sarsa_seed123.npy"
)

env = create_env()

state, _ = env.reset(seed=6000)
rng = np.random.default_rng(6000)

action_counts = Counter()
total_reward = 0
done = False

while not done:
    q_values = qtable[state]

    best_actions = np.flatnonzero(
        np.isclose(q_values, np.max(q_values))
    )

    action = int(rng.choice(best_actions))

    action_counts[action] += 1

    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

env.close()

print("\n=== EXPECTED SARSA POLICY ===")
print("Total reward:", round(total_reward, 2))
print("Actions selected:", dict(sorted(action_counts.items())))

total = sum(action_counts.values())
zero_percentage = 100 * action_counts[10] / total

print(f"Zero-torque action (10): {zero_percentage:.1f}%")
