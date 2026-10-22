
import sys
from deustorl.expected_sarsa import ExpectedSarsa
from optuna_common import run_study

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50

run_study(ExpectedSarsa, "expected_sarsa", n_trials)
