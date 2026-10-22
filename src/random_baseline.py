
import csv
from pathlib import Path

import numpy as np
from common import create_env

N_EPISODES = 50
rewards = []
results = []

env = create_env()

for episode in range(N_EPISODES):
    seed = 1000 + episode
    state, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    rewards.append(total_reward)
    results.append({
        "episode": episode + 1,
        "seed": seed,
        "reward": total_reward,
        "steps": steps,
    })

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Steps = {steps}"
    )

env.close()

print("\n=== RANDOM BASELINE ===")
print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Std reward: {np.std(rewards):.2f}")

output_dir = Path("results/tables")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "random_baseline.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["episode", "seed", "reward", "steps"]
    )
    writer.writeheader()
    writer.writerows(results)
