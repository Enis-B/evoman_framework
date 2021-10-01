'''
################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
This file is inspired by  the dummy_demo.py provided
with the EvoMan framework, as well as the example for
implementing a XOR-network from the NEAT documentation page
https://github.com/karinemiras/evoman_framework
https://neat-python.readthedocs.io/en/latest/xor_example.html
'''
from __future__ import print_function
import neat
import visualize
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from evoman.environment import Environment

# use the neat controller instead of the default demo controller
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

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# runs simulation for gain
def simulation_for_gain(env,x):
    f,p,e,t = env.play(pcont=x)
    return p,e

# dummy demo evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# non-parallel eval function
def eval(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simulation(env,net)

# parallel eval function
def eval_p(genome, config):
    # This function will run in parallel:
    # only evaluates a single genome and returns the fitness
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return simulation(env,net)

# parallel gain function
def eval_winner(winner, config):
    # This function will run in parallel:
    # only evaluates a single genome and returns the fitness
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    return simulation_for_gain(env,net)

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
        
        # Run for up to x generations in parallel.
        pe = neat.ParallelEvaluator(4, eval_p)
        winner = p.run(pe.evaluate, 20)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

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

            # Run for up to x generations in parallel.
            pe = neat.ParallelEvaluator(4, eval_p)
            winner = p.run(pe.evaluate, 50)

            # Display the winning genome.
            print('\nWinner genome:\n{!s}'.format(winner))
            real_winner = stats.best_genome()

            # Display the winning genome according to gain.
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

        # visualisation
        visualize.plot_stats_avg(avg_best,avg_avg_fitness,avg_stdev_fitness,50, ylog=False, view=True)
        fig = plt.figure(figsize =(10, 7))
        plt.title("Boxplot of mean gains")
        plt.xlabel('EA')
        plt.ylabel('mean of gains across 5 runs for each experiment')
        plt.boxplot(np.array(mean_gain_list))
        plt.savefig('boxplot_gain.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_file')
    run(config_path)

