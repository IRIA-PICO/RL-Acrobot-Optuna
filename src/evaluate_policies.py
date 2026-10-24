
import numpy as np
import pandas as pd
from pathlib import Path
from common import create_env

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"

ALGORITHMS = [
    "qlearning",
    "sarsa",
    "expected_sarsa"
]

results = []

for name in ALGORITHMS:

    qtable = np.load(MODELS / f"{name}_seed123.npy")
    env = create_env()

    rewards = []
    zero_actions = 0
    total_actions = 0

    for seed in range(5000, 5050):

        state, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        episode_reward = 0
        done = False

        while not done:

            q = qtable[state]

            # Greedy policy with random tie-breaking
            best_actions = np.flatnonzero(
                np.isclose(q, np.max(q))
            )

            action = int(rng.choice(best_actions))

            if action == 10:
                zero_actions += 1

            total_actions += 1

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    env.close()

    result = {
        "algorithm": name,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "zero_action_pct": 100 * zero_actions / total_actions
    }

    results.append(result)

    print(f"\n=== {name.upper()} ===")
    print(f"Mean reward: {result['mean_reward']:.2f}")
    print(f"Std reward: {result['std_reward']:.2f}")
    print(f"Zero Torque usage: {result['zero_action_pct']:.1f}%")

output = ROOT / "results" / "tables"
output.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    output / "seed123_evaluation.csv",
    index=False
)
