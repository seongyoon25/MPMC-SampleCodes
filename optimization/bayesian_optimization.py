"""
bayesian_optimization.py

Author: Seongyoon Kim (seongyoonk25@gmail.com)
Date: 2023-05-16

This script provides a basic implementation of Bayesian optimization using the bayes_opt library. 
It includes:

1. Definition of a fitness function for evaluating candidate solutions.
2. Configuration of parameter bounds for Bayesian optimization.
3. Setting up a logger to record the optimization progress.
4. Configuration and execution of Bayesian optimization, with options for starting a new run or resuming a previous run.
5. Display of the best solution and its fitness value.

Please make sure to install the bayes_opt library (pip install bayesian-optimization) before running this script.
"""


from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events


# funtion that returns a score (the higher, the better)
# this is an example for three parameters
def fitness_function(x0, x1, x2):
    fitness = ...
    return fitness


# set parmeter bounds
BOptimizer = BayesianOptimization(
    f=fitness_function,
    pbounds={  # parameter names with lower and upper bounds
        'x0': (..., ...),
        'x1': (..., ...),
        'x2': (..., ...),
    },
    random_state=21,
    verbose=2
    )

new_run = True
if new_run:
    # set logger
    logger = JSONLogger(path='path_to_log/BO_log.json')
    BOptimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # set manual parameter points
    BOptimizer.probe(
        params={
            'x0': ...,
            'x1': ...,
            'x2': ...,
        },
        lazy=True,
    )

    # run BO
    BOptimizer.maximize(
        init_points=10,  # random initial points
        n_iter=20,  # steps of Bayesian optimization
        alpha=1,  # increase alpha for extra flexibility
        n_restarts_optimizer=1
    )
else:
    # load
    load_logs(BOptimizer, logs=['path_to_log/BO_log.json'])

# check the results
print(BOptimizer.max)

# get items explicitly
x0, x1, x2 = [item[1] for item in BOptimizer.max['params'].items()]
