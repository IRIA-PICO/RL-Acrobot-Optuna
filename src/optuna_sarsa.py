
import sys
from deustorl.sarsa import Sarsa
from optuna_common import run_study

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50

run_study(Sarsa, "sarsa", n_trials)