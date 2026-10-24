
import csv
from pathlib import Path

import numpy as np
from common import create_env

SEEDS = range(5000, 5050)
ZERO_ACTION = 10

results = []

for policy_name in ["random", "zero_torque"]:

    env = create_env()
    rewards = []

    for seed in SEEDS:
        state, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        total_reward = 0.0
        done = False

        while not done:
            if policy_name == "random":
                action = env.action_space.sample()
            else:
                action = ZERO_ACTION

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

        results.append({
            "policy": policy_name,
            "seed": seed,
            "reward": total_reward,
        })

    env.close()

    print(f"\n=== {policy_name.upper()} ===")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")

output = Path("results/tables")
output.mkdir(parents=True, exist_ok=True)

with open(output / "final_baselines.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["policy", "seed", "reward"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nFinal baselines saved!")
