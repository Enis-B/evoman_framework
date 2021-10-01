################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
from __future__ import print_function
# import os
import neat
import visualize
import random
# from deap import base
# from deap import creator
# from deap import tools
import numpy as np
import matplotlib.pyplot as plt
import pickle

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from neat_controller import player_controller

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller = player_controller(n_hidden_neurons),
                  randomini='yes',
                  enemymode="static",
                  level=2,
                  speed="fastest")

'''
# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]
'''

'''
# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state
'''

'''
# number of weights for multilayer with 10 hidden neurons

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
'''

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# runs simulation for gain
def simulation_for_gain(env,x):
    f,p,e,t = env.play(pcont=x)
    return p,e

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))
'''
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2
'''

def eval(genomes, config):

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #evaluate(genomes)
        # print(genome_id, " genome:\n", str(genome)[:500], '\n\n\n')
        genome.fitness = simulation(env,net)


def eval_p(genome, config):
    # This function will run in parallel:
    # only evaluates a single genome and returns the fitness
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #evaluate(genomes)
    # print(genome_id, " genome:\n", str(genome)[:500], '\n\n\n')
    # genome.fitness = simulation(env,net)
    return simulation(env,net)

def eval_winner(winner, config):
    # This function will run in parallel:
    # only evaluates a single genome and returns the fitness
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    #evaluate(genomes)
    # print(genome_id, " genome:\n", str(genome)[:500], '\n\n\n')
    # genome.fitness = simulation(env,net)
    return simulation_for_gain(env,net)

'''
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
'''

'''
def main():
    pop = toolbox.population(n=300)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 100 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
'''


def run(config_file):
    run_mode = "test"
    if run_mode == "train":

        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)
        # initializes simulation in individual evolution mode, for single static enemy.
        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        #p.add_reporter(neat.Checkpointer(5))
        # Run for up to x generations.
        # winner = p.run(eval, 20)
        pe = neat.ParallelEvaluator(4, eval_p)
        winner = p.run(pe.evaluate, 20)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        '''
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(xor_inputs,xor_outputs):
            output = winner_net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
        node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
        '''

        visualize.draw_net(config, winner, True)

        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    elif run_mode == "test":
        mean_gain_list = []
        for i in range(10):
            # Load configuration.
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)
            # initializes simulation in individual evolution mode, for single static enemy.
            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            #p.add_reporter(neat.Checkpointer(5))
            # Run for up to x generations.
            # winner = p.run(eval, 20)
            pe = neat.ParallelEvaluator(4, eval_p)
            winner = p.run(pe.evaluate, 50)
            # Display the winning genome.
            print('\nWinner genome:\n{!s}'.format(winner))
            real_winner = stats.best_genome()
            # Display the winning genome.
            print('\nBest genome:\n{!s}'.format(real_winner))
            winner_gain = 0
            for cnt in range(5):
                p,e = eval_winner(real_winner,config)
                winner_gain = winner_gain + (p - e)
            mean_gain = winner_gain/5
            mean_gain_list.append(mean_gain)

            # add mean_gains to file for later stat. test
            with open('mean_gains_ea1_enemy1', 'wb') as fp:
                pickle.dump(mean_gain_list, fp)
            #with open ('mean_gains_ea1_enemy1', 'rb') as fp:
            #    itemlist = pickle.load(fp)

            '''
            # Show output of the most fit genome against training data.
            print('\nOutput:')
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            for xi, xo in zip(xor_inputs,xor_outputs):
                output = winner_net.activate(xi)
                print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
            node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
            '''

            best_fitness = [c.fitness for c in stats.most_fit_genomes]
            avg_fitness = np.array(stats.get_fitness_mean())
            stdev_fitness = np.array(stats.get_fitness_stdev())


            if i > 0:
                avg_best = [ele1 + ele2 for ele1,ele2 in zip(avg_best,best_fitness)]
                avg_avg_fitness = [ele1 + ele2 for ele1,ele2 in zip(avg_avg_fitness,avg_fitness)]
                avg_stdev_fitness = [ele1 + ele2 for ele1,ele2 in zip(avg_stdev_fitness,stdev_fitness)]
            else:
                avg_best = best_fitness
                avg_avg_fitness = avg_fitness.tolist()
                avg_stdev_fitness = stdev_fitness.tolist()


        avg_best = [number / 10 for number in avg_best]
        avg_avg_fitness = [number / 10 for number in avg_avg_fitness]
        avg_stdev_fitness = [number / 10 for number in avg_stdev_fitness]


        visualize.plot_stats_avg(avg_best,avg_avg_fitness,avg_stdev_fitness,50, ylog=False, view=True)
        fig = plt.figure(figsize =(10, 7))
        plt.title("Boxplot of mean gains")
        plt.xlabel('EA')
        plt.ylabel('mean of gains across 5 runs for each experiment')
        plt.boxplot(np.array(mean_gain_list))
        plt.savefig('boxplot_gain.png', bbox_inches='tight')
        plt.show()


    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_file')
    run(config_path)

