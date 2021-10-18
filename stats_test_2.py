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
import scipy
from scipy import stats
import pylab

with open ('dummy_gains/mean_gains_deap_enemy7,8', 'rb') as fp:
    ea1_enemy1 = pickle.load(fp)
with open ('dummy_gains/mean_gains_deap_enemy2,5,6', 'rb') as fp:
    ea1_enemy2 = pickle.load(fp)
with open ('dummy_gains/mean_gains_neat_enemy7,8', 'rb') as fp:
    ea2_enemy1 = pickle.load(fp)
with open ('dummy_gains/mean_gains_deap_enemy2,5,6', 'rb') as fp:
    ea2_enemy2 = pickle.load(fp)


stats.probplot(ea1_enemy1, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea1_enemy2, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea2_enemy1, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea2_enemy2, dist="norm", plot=pylab)
pylab.show()

print("Enemy 1")
print(scipy.stats.ks_2samp(ea1_enemy1,ea2_enemy1))
print("Enemy 2")
print(scipy.stats.ks_2samp(ea1_enemy2,ea2_enemy2))

print("Enemy 1")
print(scipy.stats.ttest_ind(ea1_enemy1,ea2_enemy1))
print("Enemy 2")
print(scipy.stats.ttest_ind(ea1_enemy2,ea2_enemy2))