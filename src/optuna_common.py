
import random
from pathlib import Path

import numpy as np
import optuna
from optuna.trial import TrialState

from deustorl.common import EpsilonGreedyPolicy
from common import create_env


TRAINING_STEPS = 30_000
TRAIN_SEED = 42
VALIDATION_SEEDS = range(3000, 3020)
MAX_TRIALS = 50


def evaluate(agent):
    env = create_env()
    rng = np.random.default_rng(2026)
    rewards = []

    for seed in VALIDATION_SEEDS:
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        while not done:
            q = np.asarray(
                agent.q_table[state], dtype=float
            )

            best_actions = np.flatnonzero(
                np.isclose(q, np.max(q))
            )
            action = int(rng.choice(best_actions))

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    env.close()
    return float(np.mean(rewards))


def run_study(algorithm_class, name, requested_trials=50):

    output = Path("results/optuna").resolve()
    output.mkdir(parents=True, exist_ok=True)

    db_path = output / f"{name}.db"

    def objective(trial):

        lr = trial.suggest_float(
            "learning_rate", 0.01, 0.5, log=True
        )
        gamma = trial.suggest_float(
            "discount_rate", 0.80, 0.999
        )
        epsilon = trial.suggest_float(
            "epsilon", 0.05, 0.5
        )
        lrdecay = trial.suggest_float(
            "lrdecay", 0.90, 1.0
        )

        random.seed(TRAIN_SEED)
        np.random.seed(TRAIN_SEED)

        env = create_env()
        env.reset(seed=TRAIN_SEED)
        env.action_space.seed(TRAIN_SEED)

        agent = algorithm_class(env)
        policy = EpsilonGreedyPolicy(epsilon=epsilon)

        agent.learn(
            policy=policy,
            n_steps=TRAINING_STEPS,
            discount_rate=gamma,
            lr=lr,
            lrdecay=lrdecay,
            n_episodes_decay=100,
            tb_episode_period=1000,
        )

        env.close()

        mean_reward = evaluate(agent)

        print(
            f"Trial {trial.number}: "
            f"Mean reward = {mean_reward:.2f}"
        )

        return mean_reward

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    completed = len(
        study.get_trials(
            states=(TrialState.COMPLETE,)
        )
    )

    remaining = max(0, MAX_TRIALS - completed)

    study.optimize(
        objective,
        n_trials=min(requested_trials, remaining)
    )

    if study.best_trial:
        print("\n=== BEST RESULT ===")
        print("Best reward:", study.best_value)
        print("Best parameters:", study.best_params)
        print("Database:", db_path)
