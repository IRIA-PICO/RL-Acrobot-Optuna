
import csv
import random
from pathlib import Path

import numpy as np
import optuna

from common import create_env
from deustorl.common import EpsilonGreedyPolicy
from deustorl.qlearning import QLearning
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa


ROOT = Path(__file__).resolve().parents[1]
TRAINING_STEPS = 200_000
TRAIN_SEED = 42
EVAL_SEEDS = range(5000, 5050)

ALGORITHMS = {
    "qlearning": QLearning,
    "sarsa": Sarsa,
    "expected_sarsa": ExpectedSarsa,
}


def evaluate(agent, seeds):
    env = create_env()
    results = []

    for seed in seeds:
        state, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        total_reward = 0.0
        done = False

        while not done:
            q = np.asarray(agent.q_table[state], dtype=float)
            best_actions = np.flatnonzero(
                np.isclose(q, np.max(q))
            )
            action = int(rng.choice(best_actions))

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        results.append({
            "seed": seed,
            "reward": total_reward
        })

    env.close()
    return results


all_results = []

for name, Algorithm in ALGORITHMS.items():

    # Load best parameters from Optuna
    db_path = ROOT / "results" / "optuna" / f"{name}.db"

    study = optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{db_path}"
    )

    params = study.best_params

    print(f"\n=== LONG TRAINING: {name.upper()} ===")
    print("Hyperparameters:", params)

    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    env = create_env()
    env.reset(seed=TRAIN_SEED)
    env.action_space.seed(TRAIN_SEED)

    agent = Algorithm(env)

    policy = EpsilonGreedyPolicy(
        epsilon=params["epsilon"]
    )

    agent.learn(
        policy=policy,
        n_steps=TRAINING_STEPS,
        discount_rate=params["discount_rate"],
        lr=params["learning_rate"],
        lrdecay=params["lrdecay"],
        n_episodes_decay=100,
        tb_episode_period=20,
    )

    env.close()

    # Evaluate on unseen seeds
    evaluation = evaluate(agent, EVAL_SEEDS)

    rewards = np.array([
        row["reward"] for row in evaluation
    ])

    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Std reward: {rewards.std():.2f}")

    for row in evaluation:
        all_results.append({
            "algorithm": name,
            **row
        })


# Save final evaluation results
output = ROOT / "results" / "tables"
output.mkdir(parents=True, exist_ok=True)

with open(output / "long_training_evaluation.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["algorithm", "seed", "reward"]
    )
    writer.writeheader()
    writer.writerows(all_results)

print("\nAll long trainings completed!")
