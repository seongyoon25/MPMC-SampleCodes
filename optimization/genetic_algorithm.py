"""
genetic_algorithm.py

Author: Seongyoon Kim (seongyoonk25@gmail.com)
Date: 2023-05-16

This script provides a basic implementation of a genetic algorithm using the PyGAD library.
It includes:

1. Definition of a score function for evaluating candidate solutions.
2. Implementation of the fitness function required by the PyGAD library.
3. Configuration of the parameter bounds and gene space for the genetic algorithm.
4. Definition of an on_generation callback function to display progress during optimization.
5. Configuration and execution of the genetic algorithm using PyGAD.
6. Display of the best solution and its fitness value.

Please make sure to install the PyGAD library (pip install pygad) before running this script.
"""


import pygad


# funtion that returns a score (the higher, the better)
# this is an example for three parameters
def score_function(x0, x1, x2):
    score = ...
    return score


# ga function in pygad allows only two arguments: `solution`, `solution_idx`
def fitness_func(solution, solution_idx):
    fitness = score_function(*solution)
    return fitness


# parameter names with lower and upper bounds
pbounds = {
    'x0': (..., ...),
    'x1': (..., ...),
    'x2': (..., ...),
    }
gene_space = []
for par in pbounds.keys():
    gene_space.append({'low': pbounds[par][0], 'high': pbounds[par][1]})


def on_generation(ga):
    print('Generation {} | Max fitness {:.4f}'.format(ga.generations_completed, max(ga.best_solutions_fitness)))


# set GA parameters
ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       num_genes=len(gene_space),
                       sol_per_pop=10*len(gene_space),
                       mutation_percent_genes=10,
                       gene_space=gene_space,
                       stop_criteria='saturate_10',  # stop when 10 step congestion
                       suppress_warnings=True,
                       save_solutions=False,
                       save_best_solutions=True,
                       on_generation=on_generation
                       )

# run GA
ga_instance.run()

# plot fitness
ga_instance.plot_fitness()

# get best results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution)
print('Fitness value of the best solution = {solution_fitness}'.format(solution_fitness=solution_fitness))
print('Index of the best solution : {solution_idx}'.format(solution_idx=solution_idx))

# get items explicitly
x0, x1, x2 = solution
