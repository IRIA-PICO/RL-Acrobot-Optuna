
import gymnasium as gym
import numpy as np

from deustorl.helpers import (
    DiscretizedObservationWrapper,
    DiscretizedActionWrapper,
)

# Initial discretization configuration
N_BINS_OBS = 3
N_BINS_ACTIONS = 3

# Proposed observation bounds for Reacher-v5
OBS_LOW = np.array([
    -1, -1, -1, -1,     # cos/sin of joint angles
    -0.25, -0.25,       # target position
    -10, -10,           # joint velocities
    -0.5, -0.5          # fingertip-target vector
], dtype=np.float64)

OBS_HIGH = -OBS_LOW

# Original environment
env = gym.make("Reacher-v5")

print("=== ORIGINAL ENVIRONMENT ===")
print("Observations:", env.observation_space)
print("Actions:", env.action_space)

# Apply the professor's wrappers
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

print("\n=== DISCRETIZED ENVIRONMENT ===")
print("Observations:", env.observation_space)
print("Actions:", env.action_space)

# Test reset and steps
state, info = env.reset(seed=42)
env.action_space.seed(42)

print("\nInitial state:", state)

for step in range(5):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1}: "
        f"State={next_state}, "
        f"Action={action}, "
        f"Reward={reward:.3f}"
    )

    if terminated or truncated:
        break

env.close()
print("\nTest completed!")

