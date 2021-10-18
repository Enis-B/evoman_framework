from __future__ import print_function
import neat
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import visualize

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_deap_0'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  enemies=[2,5,6], # (1,5) (2,6) (4,6,7)
                  playermode="ai",
                  player_controller = player_controller(n_hidden_neurons),
                  multiplemode="yes",
                  randomini='yes',
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

# number of weights for multilayer with 10 hidden neurons. (265)
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f, 1. / len(x) # DEAP GA wants the fitness as a tuple

# runs simulation for gain
def sim_for_gain(env,x):
    f,p,e,t = env.play(pcont=x)
    return p,e
'''
# dummy demo evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))
'''
def evaluate(x):
    return simulation(env,x)

def eval_best(x,env):
    return sim_for_gain(env,x)

## RUN MODE
run_mode = 'test'

if run_mode == 'train':
    print("Run Mode: ",run_mode)
    env.state_to_log() # checks environment state
    mean_gain_list = []
    ini = time.time()  # sets time marker
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    #N_vars
    IND_SIZE=n_vars

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=IND_SIZE)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) #0.1
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)


    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=499,
                                   stats=stats, halloffame=hof, verbose=True)


    fim = time.time() # prints total execution time
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

    env.state_to_log() # checks environment state

    print('\nLog\n',log,'\n',log.select('avg'))
    #print('\nPop\n',pop,len(pop))
    #print('\nHoF\n',hof,len(hof[0]))
    print('\nHof fitness\n',hof[-1].fitness.values[0])

    winner_gain = 0
    for opponent in range(1,9):
        experiment_name = "dummy_deap"+"_0"+"/all_enemies"+str(opponent)+"/"
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        env_single = Environment(experiment_name=experiment_name,
                                 enemies=[opponent],
                                 playermode="ai",
                                 player_controller = player_controller(n_hidden_neurons),
                                 randomini='yes',
                                 enemymode="static",
                                 level=2,
                                 speed="fastest",
                                 logs='on',
                                 savelogs='yes'
                                 )
        for cnt in range(5):
            p,e = eval_best(hof[0],env_single)
            winner_gain = winner_gain + (p - e)

        mean_gain = winner_gain/5
        mean_gain_list.append(mean_gain)

    # add mean_gains to file for later stat. test
    with open('test_deap_task2/enemy2,5,6_deap_0,5xover_0.1indpb_3tournsize_experiment/mean_gains_ea2_enemy2,5,6', 'wb') as fp:
        pickle.dump(mean_gain_list, fp)
    #with open ('mean_gains_ea1_enemy1', 'rb') as fp:
    #    itemlist = pickle.load(fp)

    #print('\nLog\n',log,'\n',log.select('avg')) ## average fitness each gen
    #print('\nPop\n',pop,'\n',len(pop)) ## final population
    print('\nHoF\n',hof,'\n',len(hof[0])) ## best genome seen

    a_file = open(str(int(hof[-1].fitness.values[0]))+"_"+".txt", "w")
    np.savetxt(a_file, np.array(hof[0]))
    a_file.close()

    fig = plt.figure(figsize =(10, 7))
    plt.title("Boxplot of mean gains")
    plt.xlabel('EA')
    plt.ylabel('mean of gains across 5 runs for each experiment')
    plt.boxplot(np.array(mean_gain_list))
    plt.savefig('boxplot_gain_train.png', bbox_inches='tight')
    plt.show()

    gens = 500
    best=log.select('max')
    avg=log.select('avg')
    std=log.select('std')

    best_std = np.std(np.array(best))
    avg_std = np.std(np.array(avg))

    visualize.plot_stats_deap_avg(best,avg,std,gens,best_std,avg_std,ylog=False, view=True)

elif run_mode == 'test':
    print("Run Mode: ",run_mode)
    ini_total = time.time()  # sets time marker
    mean_gain_list = []
    for i in range(10):
        experiment_name = 'dummy_deap' +"_"+ str(i+1)
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        env.state_to_log() # checks environment state
        ini = time.time()  # sets time marker

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        #N_vars
        IND_SIZE=n_vars

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=IND_SIZE)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        ## Operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) #0.05
        toolbox.register("select", tools.selTournament,tournsize=3) #5
        toolbox.register("evaluate", evaluate)


        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=1.0, mutpb=0.2, ngen=39, #0.5 , 0.2
                                       stats=stats, halloffame=hof, verbose=True)

        winner_gain = 0
        for opponent in range(1,9):
            experiment_name = "dummy_deap"+"_"+str(i)+"/all_enemies"+str(opponent)+"/"
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)
            env_single = Environment(experiment_name=experiment_name,
                                     enemies=[opponent],
                                     playermode="ai",
                                     player_controller = player_controller(n_hidden_neurons),
                                     randomini='yes',
                                     enemymode="static",
                                     level=2,
                                     speed="fastest",
                                     logs='on',
                                     savelogs='yes'
                                     )
            for cnt in range(5):
                p,e = eval_best(hof[0],env_single)
                winner_gain = winner_gain + (p - e)

            mean_gain = winner_gain/5
            mean_gain_list.append(mean_gain)

        # add mean_gains to file for later stat. test
        with open('deap_256/mean_gains_ea2_enemy2,5,6', 'wb') as fp:
            pickle.dump(mean_gain_list, fp)
        #with open ('mean_gains_ea1_enemy1', 'rb') as fp:
        #    itemlist = pickle.load(fp)

        #print('\nLog\n',log,'\n',log.select('avg')) ## average fitness each gen
        #print('\nPop\n',pop,'\n',len(pop)) ## final population
        print('\nHoF\n',hof,'\n',len(hof[0])) ## best genome seen

        a_file = open(str(int(hof[-1].fitness.values[0]))+"_"+str(i)+".txt", "w")
        np.savetxt(a_file, np.array(hof[0]))
        a_file.close()


        best=log.select('max')
        avg=log.select('avg')
        std=log.select('std')
        ## single run plot
        gens = 40
        visualize.plot_stats_deap(best,avg,std,gens)

        if i > 0:
            avg_best = [ele1 + ele2 for ele1,ele2 in zip(avg_best,best)]
            avg_avg_fitness = [ele1 + ele2 for ele1,ele2 in zip(avg_avg_fitness,avg)]
            avg_stdev_fitness = [ele1 + ele2 for ele1,ele2 in zip(avg_stdev_fitness,std)]
        else:
            avg_best = best
            avg_avg_fitness = avg
            avg_stdev_fitness = std

        env.state_to_log() # checks environment state
        fim = time.time() # prints execution time
        print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')

    avg_best = [number / 10 for number in avg_best]
    avg_avg_fitness = [number / 10 for number in avg_avg_fitness]
    avg_stdev_fitness = [number / 10 for number in avg_stdev_fitness]

    best_std = np.std(np.array(avg_best))
    avg_std = np.std(np.array(avg_avg_fitness))

    fim_total = time.time() # prints execution time
    print( '\nExecution time: '+str(round((fim_total-ini_total)/60))+' minutes \n')

    gens = 40
    # visualisation
    visualize.plot_stats_deap_avg(avg_best,avg_avg_fitness,avg_stdev_fitness,gens,best_std,avg_std,ylog=False, view=True)
    fig = plt.figure(figsize =(10, 7))
    plt.title("Boxplot of mean gains")
    plt.xlabel('EA')
    plt.ylabel('mean of gains across 5 runs for each experiment')
    plt.boxplot(np.array(mean_gain_list))
    plt.savefig('boxplot_gain.png', bbox_inches='tight')
    plt.show()