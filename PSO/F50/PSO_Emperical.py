import random
import string
import time

from indago import PSO
#from pyswarm import pso
from ObjectiveFunction import objective
from Ac_constants import AcConstants as AC

AC = AC()
#NO AVL skip due to constraints

def generate_random_code(length):
    # Choose from uppercase letters and digits
    characters = string.ascii_uppercase + string.digits + string.ascii_lowercase
    random_code = ''.join(random.choice(characters) for _ in range(length))
    return random_code


if __name__ == '__main__':
    start_time = time.time()


    file_name_base = 'log_file_' + AC.aircraft_config
    random_code = generate_random_code(4)
    file_name = file_name_base +'_'+ random_code
    #
    # while os.path.isfile(file_path):
    #     i += 1
    #     file_name = file_name_base + f'{i}'
    #     file_path = file_name + '.txt'
    #     print('new filepath:', file_path)



    optimizer = PSO()
    optimizer.ub = AC.upper_bound
    optimizer.lb = AC.lower_bound
    optimizer.evaluation_function = objective

    optimizer.objectives = 1
    optimizer.objective_labels = ['Highest CL/CD']
    optimizer.constraints = 9
    optimizer.constraint_labels = ['upperbound kink location','lowerbound kink location','root chord','tip chord','lowerbound kink chord','upperbound wing weight','lowerbound wing weight','lowerbound fuel volume','Wing Loading']
    optimizer.params['inertia'] = 'anakatabatic'
    optimizer.params['akb_model'] = 'Languid'  # other options explained below
    optimizer.params['cognitive_rate'] = 1.0  # PSO parameter also known as c1 (should range from 0.0 to 2.0); default cognitive_rate=1.0
    optimizer.params['social_rate'] = 1.0  # PSO parameter also known as c2 (should range from 0.0 to 2.0); default social_rate=1.0
    optimizer.params['swarm_size'] = 100
    optimizer.max_iterations = 1000  # optional maximum allowed number of method iterations; when surpassed, optimization is stopped (if reached before other stopping conditions are satistifed)
    optimizer.max_evaluations = 7500  # optional maximum allowed number of function evaluations; when surpassed, optimization is stopped (if reached before other stopping conditions are satistifed); if no stopping criteria given, default max_evaluations=50*dimensions**2
    optimizer.target_fitness = -41 # optional fitness threshold; when reached, optimization is stopped (if it didn't already stop due to exhausted pso.max_iterations or pso.maximum_evaluations)
    optimizer.monitoring = 'basic'
    optimizer.convergence_log_file  = 'Results/' + file_name
    optimizer.population, optimizer.generation, optimizer.fitness_criterion = 20, 40,  0.01



    result = optimizer.optimize()
    print('Swarm size: ',optimizer.params['swarm_size'] )
    print('D_cruise', -result.f, "Best design", result.X)
    print("--- %s seconds ---" % (time.time() - start_time))