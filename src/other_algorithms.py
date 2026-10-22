
import csv
import inspect
import random
from pathlib import Path

import numpy as np

from common import create_env
from deustorl.common import EpsilonGreedyPolicy
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa

ALGORITHMS = {
    "sarsa": Sarsa,
    "expected_sarsa": ExpectedSarsa
}

N_STEPS = 30_000
N_EVAL = 50
SEED = 42

OUTPUT = Path("results/tables")
OUTPUT.mkdir(parents=True, exist_ok=True)

for name, Algorithm in ALGORITHMS.items():

    print(f"\n=== TRAINING {name.upper()} ===")

    random.seed(SEED)
    np.random.seed(SEED)

    env = create_env()
    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    agent = Algorithm(env)
    policy = EpsilonGreedyPolicy(epsilon=0.2)

    config = {
        "n_steps": N_STEPS,
        "discount_rate": 0.95,
        "lr": 0.1,
        "lrdecay": 1.0,
        "tb_episode_period": 20,
    }

    # Use only parameters accepted by the class
    params = inspect.signature(agent.learn).parameters
    config = {k: v for k, v in config.items() if k in params}

    agent.learn(policy, **config)
    env.close()

    print(f"\n=== EVALUATING {name.upper()} ===")

    eval_env = create_env()
    rng = np.random.default_rng(SEED)
    rewards = []
    results = []

    for episode in range(N_EVAL):

        seed = 1000 + episode
        state, _ = eval_env.reset(seed=seed)

        total_reward = 0.0
        done = False

        while not done:

            q_values = np.asarray(
                agent.q_table[state], dtype=float
            )

            # Greedy action with random tie-breaking
            best_actions = np.flatnonzero(
                np.isclose(q_values, np.max(q_values))
            )
            action = int(rng.choice(best_actions))

            state, reward, terminated, truncated, _ = (
                eval_env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        results.append({
            "episode": episode + 1,
            "seed": seed,
            "reward": total_reward
        })

    eval_env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"vs Random: {mean_reward - (-44.58):+.2f}")
    print(f"vs Zero torque: {mean_reward - (-11.68):+.2f}")

    with open(OUTPUT / f"{name}_pilot.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "seed", "reward"]
        )
        writer.writeheader()
        writer.writerows(results)

print("\nAll pilot experiments completed!")
