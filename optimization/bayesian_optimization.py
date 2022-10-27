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
