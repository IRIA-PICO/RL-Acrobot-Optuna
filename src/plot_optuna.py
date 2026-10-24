
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "results" / "optuna"
OUTPUT = ROOT / "results" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

names = ["qlearning", "sarsa", "expected_sarsa"]
labels = ["Q-Learning", "SARSA", "Expected SARSA"]

studies = {}

for name in names:
    db = (DB_DIR / f"{name}.db").resolve()
    studies[name] = optuna.load_study(
        study_name=name,
        storage=f"sqlite:///{db}"
    )

# ----------------------------------------
# 1. OPTIMIZATION HISTORY
# ----------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                         sharey=True)

for ax, name, label in zip(axes, names, labels):
    trials = [
        t for t in studies[name].trials
        if t.state == TrialState.COMPLETE
    ]

    x = [t.number for t in trials]
    y = [t.value for t in trials]
    best = np.maximum.accumulate(y)

    ax.scatter(x, y, alpha=0.55, label="Trial reward")
    ax.plot(x, best, linewidth=2.5,
            label="Best so far")

    ax.set_title(label)
    ax.set_xlabel("Trial")
    ax.grid(alpha=0.25)
    ax.legend()

axes[0].set_ylabel("Mean validation reward")
fig.suptitle("Optuna Optimization History")
fig.tight_layout()
fig.savefig(OUTPUT / "01_optimization_history.png",
            dpi=250, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------
# 2. HYPERPARAMETER IMPORTANCE
# ----------------------------------------

importances = {
    name: get_param_importances(studies[name])
    for name in names
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                         sharex=True)

params = [
    "learning_rate",
    "discount_rate",
    "epsilon",
    "lrdecay"
]

for ax, name, label in zip(axes, names, labels):
    values = [importances[name].get(p, 0) for p in params]

    ax.barh(params, values)
    ax.set_title(label)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    for i, value in enumerate(values):
        ax.text(value + 0.01, i, f"{value:.2f}",
                va="center", fontsize=9)

fig.suptitle("Hyperparameter Importance Comparison")
fig.tight_layout()
fig.savefig(OUTPUT / "02_parameter_importance.png",
            dpi=250, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------
# 3. MOST IMPORTANT PARAMETER VS REWARD
# ----------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5),
                         sharey=True)

for ax, name, label in zip(axes, names, labels):
    study = studies[name]

    # Select the most important parameter
    parameter = max(importances[name],
                    key=importances[name].get)

    trials = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
    ]

    x = [t.params[parameter] for t in trials]
    y = [t.value for t in trials]

    ax.scatter(x, y, alpha=0.7)

    # Highlight best trial
    ax.scatter(
        study.best_params[parameter],
        study.best_value,
        marker="*",
        s=200,
        label="Best trial"
    )

    if parameter == "learning_rate":
        ax.set_xscale("log")

    ax.set_title(f"{label}\nMost important: {parameter}")
    ax.set_xlabel(parameter)
    ax.grid(alpha=0.25)
    ax.legend()

axes[0].set_ylabel("Mean validation reward")

fig.suptitle("Hyperparameters and Performance")
fig.tight_layout()
fig.savefig(OUTPUT / "03_parameter_vs_reward.png",
            dpi=250, bbox_inches="tight")
plt.close(fig)

print("Three figures successfully saved!")

