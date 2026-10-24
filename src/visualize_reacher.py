
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import optuna

from common import create_env
from deustorl.common import EpsilonGreedyPolicy
from deustorl.qlearning import QLearning
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"
VIDEOS = ROOT / "results" / "videos"

MODELS.mkdir(parents=True, exist_ok=True)
VIDEOS.mkdir(parents=True, exist_ok=True)

TRAIN_SEED = 123
VISUALIZATION_SEED = 6000
TRAINING_STEPS = 200_000

ALGORITHMS = {
    "qlearning": QLearning,
    "sarsa": Sarsa,
    "expected_sarsa": ExpectedSarsa,
}


def get_qtable(name, Algorithm):
    path = MODELS / f"{name}_seed{TRAIN_SEED}.npy"

    if path.exists():
        print(f"Loading saved policy: {name}")
        return np.load(path)

    db = ROOT / "results" / "optuna" / f"{name}.db"

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

    print(f"\nTraining {name}...")

    agent.learn(
        policy=policy,
        n_steps=TRAINING_STEPS,
        discount_rate=params["discount_rate"],
        lr=params["learning_rate"],
        lrdecay=params["lrdecay"],
        n_episodes_decay=100,
        tb_episode_period=20,
    )

    qtable = np.asarray(
        [agent.q_table[s]
         for s in range(env.observation_space.n)],
        dtype=np.float32
    )

    np.save(path, qtable)
    env.close()

    return qtable


def record_episode(name, qtable=None):
    env = create_env(render_mode="rgb_array")

    state, _ = env.reset(seed=VISUALIZATION_SEED)
    env.action_space.seed(VISUALIZATION_SEED)

    rng = np.random.default_rng(VISUALIZATION_SEED)

    frames = [env.render()]
    total_reward = 0.0

    for step in range(50):

        if name == "random":
            action = env.action_space.sample()

        elif name == "zero_torque":
            action = 10

        else:
            values = qtable[state]
            best_actions = np.flatnonzero(
                np.isclose(values, values.max())
            )
            action = int(rng.choice(best_actions))

        state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        total_reward += reward
        frames.append(env.render())

        if terminated or truncated:
            break

    output = VIDEOS / f"{name}.gif"

    imageio.mimsave(
        output,
        frames,
        duration=0.08,
        loop=0
    )

    env.close()

    print(f"{name}: Reward = {total_reward:.2f}")
    print(f"Video saved: {output}")


# Record the two baselines
record_episode("random")
record_episode("zero_torque")

# Train/load and record learned policies
for name, Algorithm in ALGORITHMS.items():
    qtable = get_qtable(name, Algorithm)
    record_episode(name, qtable)

print("\nAll videos created!")
