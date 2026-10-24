
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

TRAIN_SEED = 2026
TRAINING_STEPS = 200_000
EVAL_SEEDS = range(5000, 5050)

ALGORITHMS = {
    "qlearning": QLearning,
    "sarsa": Sarsa,
    "expected_sarsa": ExpectedSarsa
}

results = []

for name, Algorithm in ALGORITHMS.items():

    print(f"\n=== TRAINING {name.upper()} ===")

    # Load Optuna's best hyperparameters
    db = (ROOT / "results" / "optuna" / f"{name}.db").resolve()

    study = optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{db}"
    )

    params = study.best_params

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

    # Save the trained Q-table
    models_dir = ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    qtable = np.asarray(
        [agent.q_table[s]
         for s in range(env.observation_space.n)],
        dtype=np.float32
    )

    np.save(
        models_dir / f"{name}_seed2026.npy",
        qtable
    )

    env.close()

    # Evaluate on the same 50 seeds
    eval_env = create_env()
    rewards = []
    zero_actions = 0
    total_actions = 0

    for seed in EVAL_SEEDS:
        state, _ = eval_env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        total_reward = 0.0
        done = False

        while not done:
            q = qtable[state]

            best_actions = np.flatnonzero(
                np.isclose(q, np.max(q))
            )
            action = int(rng.choice(best_actions))

            if action == 10:
                zero_actions += 1

            total_actions += 1

            state, reward, terminated, truncated, _ = (
                eval_env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    eval_env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    zero_pct = 100 * zero_actions / total_actions

    print(f"\n=== RESULTS {name.upper()} ===")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Zero Torque Usage: {zero_pct:.1f}%")

    results.append({
        "algorithm": name,
        "training_seed": TRAIN_SEED,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "zero_torque_pct": zero_pct
    })

# Save results
output = ROOT / "results" / "tables"
output.mkdir(parents=True, exist_ok=True)

with open(output / "seed2026_evaluation.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\nThird training repetition completed!")
