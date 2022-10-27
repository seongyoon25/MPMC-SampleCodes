import pybamm
import pandas as pd
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time


# funtion that returns a score (the higher, the better)
# this is an example for three parameters
def fitness_function(x0, x1, x2):
    fitness = ...
    return fitness


BOptimizer = BayesianOptimization(
    f = fitness_function,
    pbounds={  # parameter names with lower and upper bounds
        'x0': (..., ...),
        'x1': (..., ...),
        'x2': (..., ...),
        },
    random_state=21,
    verbose=2
    )

BOptimizer.maximize(
    init_points=10,  # random initial points
    n_iter=20,  # steps of Bayesian optimization
    alpha=1,
)
