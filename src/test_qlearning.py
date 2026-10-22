
import csv
import random
from pathlib import Path

import numpy as np

from common import create_env
from deustorl.common import EpsilonGreedyPolicy
from deustorl.qlearning import QLearning


# ========================================
# 1. CONFIGURATION
# ========================================

TRAINING_STEPS = 30_000
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95
EPSILON = 0.2
SEED = 42
N_EVAL_EPISODES = 50


# ========================================
# 2. TRAINING
# ========================================

np.random.seed(SEED)
random.seed(SEED)

env = create_env()

env.reset(seed=SEED)
env.action_space.seed(SEED)

agent = QLearning(env)

policy = EpsilonGreedyPolicy(epsilon=EPSILON)

print("=== TRAINING Q-LEARNING ===")

agent.learn(
    policy=policy,
    n_steps=TRAINING_STEPS,
    discount_rate=DISCOUNT_RATE,
    lr=LEARNING_RATE,
    lrdecay=1.0,
    tb_episode_period=20,
)

env.close()

print("Training completed!")


# ========================================
# 3. EVALUATION
# ========================================

eval_env = create_env()

action_counts = np.zeros(eval_env.action_space.n, dtype=int)
zero_q_steps = 0

rng = np.random.default_rng(SEED)

rewards = []
results = []

print("\n=== EVALUATING Q-LEARNING ===")

for episode in range(N_EVAL_EPISODES):

    seed = 1000 + episode
    state, info = eval_env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:

        # Greedy policy with random tie-breaking
        q_values = np.asarray(
            agent.q_table[state],
            dtype=float
        )

        best_actions = np.flatnonzero(
            np.isclose(q_values, np.max(q_values))
        )

        action = int(rng.choice(best_actions))

        
        action_counts[action] += 1

        # States whose Q-values are still all zero
        if np.all(q_values == 0):
            zero_q_steps += 1

        state, reward, terminated, truncated, info = (
            eval_env.step(action)
        )

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
        f"Reward = {total_reward:.2f}"
    )

eval_env.close()


# ========================================
# 4. RESULTS
# ========================================

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print("\n=== Q-LEARNING RESULTS ===")
print(f"Mean reward: {mean_reward:.2f}")
print(f"Std reward: {std_reward:.2f}")

# Compare with the saved random baseline
baseline_file = Path("results/tables/random_baseline.csv")

if baseline_file.exists():

    with open(baseline_file, newline="") as f:
        baseline_rows = list(csv.DictReader(f))

    baseline_mean = np.mean([
        float(row["reward"]) for row in baseline_rows
    ])

    print(f"Random baseline: {baseline_mean:.2f}")
    print(f"Improvement: {mean_reward - baseline_mean:+.2f}")

# Save evaluation
output = Path("results/tables")
output.mkdir(parents=True, exist_ok=True)

with open(output / "qlearning_pilot.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["episode", "seed", "reward", "steps"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nResults saved!")


print("\n=== POLICY ANALYSIS ===")

total_actions = action_counts.sum()

for action, count in enumerate(action_counts):
    percentage = 100 * count / total_actions
    print(f"Action {action}: {percentage:.1f}%")

print(
    f"\nZero-torque action (10): "
    f"{100 * action_counts[10] / total_actions:.1f}%"
)

print(
    f"Steps with all Q-values equal to zero: "
    f"{100 * zero_q_steps / total_actions:.1f}%"
)
