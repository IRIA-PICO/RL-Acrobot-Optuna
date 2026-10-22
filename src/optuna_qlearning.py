
import sys
from deustorl.qlearning import QLearning
from optuna_common import run_study

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50

run_study(QLearning, "qlearning", n_trials)
