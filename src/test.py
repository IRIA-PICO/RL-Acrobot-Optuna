
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np

from common import create_env

VALIDATION_SEEDS = range(1000, 1050)
TEST_SEEDS = range(2000, 2050)


def evaluate(env, action, seeds):
    rewards = []

    for seed in seeds:
        state, info = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        while not done:
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return float(np.mean(rewards)), float(np.std(rewards))


env = create_env()
results = []

print("=== FIXED ACTIONS: VALIDATION ===")

for action in range(env.action_space.n):
    continuous_action = env.env.action(action)

    mean, std = evaluate(
        env, action, VALIDATION_SEEDS
    )

    results.append({
        "action": action,
        "mean_reward": mean,
        "std_reward": std,
    })

    print(
        f"Action {action} {continuous_action}: "
        f"Mean = {mean:.2f}, Std = {std:.2f}"
    )

# Select the best fixed action using validation episodes
best = max(results, key=lambda row: row["mean_reward"])

print("\n=== BEST FIXED ACTION ===")
print("Action:", best["action"])
print("Validation mean:", round(best["mean_reward"], 2))

# Evaluate on different seeds
test_mean, test_std = evaluate(
    env, best["action"], TEST_SEEDS
)

print("Test mean:", round(test_mean, 2))
print("Test std:", round(test_std, 2))

env.close()

# True zero-torque reference in the original environment
raw_env = gym.make("Reacher-v5")

zero_action = np.zeros(2, dtype=np.float32)

zero_mean, zero_std = evaluate(
    raw_env, zero_action, TEST_SEEDS
)

print("\n=== TRUE ZERO-TORQUE BASELINE ===")
print("Action:", zero_action)
print("Test mean:", round(zero_mean, 2))
print("Test std:", round(zero_std, 2))

raw_env.close()

# Save validation results
output = Path("results/tables")
output.mkdir(parents=True, exist_ok=True)

with open(output / "fixed_action_validation.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)
