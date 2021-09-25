import random
import numpy as np
import math



#Simulated annealing algorithm
def sa_alg(high_fit, t_n, i_max): # input the highest fitness treshold, how much t_n we survey, and max iterations, i_max

    highest_fitness = high_fit
    cost_highest = high_fit.length()
    costs = [cost_highest]
    for i in range(i_max):
        fitness_temp = highest_fitness.alternative()
        u = random.random()
        cost_temp = fitness_temp.length()
        t = t_n(i)
        if cost_temp < cost_highest:
            highest_fitness = fitness_temp
            cost_highest = cost_temp
            costs.append(cost_highest)
            continue
        if t > 0:
            alpha = max(math.exp(-(fitness_temp.length()-highest_fitness.length())/t),1)
            if random.random()>= alpha:
                highest_fitness = fitness_temp
                cost_highest = cost_temp
        costs.append(cost_highest)
    return highest_fitness, np.array(costs)

