
import gymnasium as gym
import numpy as np

from deustorl.helpers import (
    DiscretizedObservationWrapper,
    DiscretizedActionWrapper,
)

ENV_NAME = "Reacher-v5"

N_BINS_OBS = 3
N_BINS_ACTIONS = 4

OBS_LOW = np.array([
    -1, -1, -1, -1,
    -0.25, -0.25,
    -10, -10,
    -0.5, -0.5
], dtype=np.float64)

OBS_HIGH = -OBS_LOW


def create_env(render_mode=None):
    env = gym.make(ENV_NAME, render_mode=render_mode)

    env = DiscretizedActionWrapper(
        env,
        n_bins=N_BINS_ACTIONS
    )

    env = DiscretizedObservationWrapper(
        env,
        n_bins=N_BINS_OBS,
        low=OBS_LOW,
        high=OBS_HIGH
    )

    return env
