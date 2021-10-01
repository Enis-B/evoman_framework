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

with open ('mean_gains_ea1_enemy1', 'rb') as fp:
    ea1_enemy1 = pickle.load(fp)
with open ('mean_gains_ea1_enemy2', 'rb') as fp:
    ea1_enemy2 = pickle.load(fp)
with open ('mean_gains_ea1_enemy3', 'rb') as fp:
    ea1_enemy3 = pickle.load(fp)
with open ('mean_gains_ea2_enemy1', 'rb') as fp:
    ea2_enemy1 = pickle.load(fp)
with open ('mean_gains_ea2_enemy2', 'rb') as fp:
    ea2_enemy2 = pickle.load(fp)
with open ('mean_gains_ea2_enemy3', 'rb') as fp:
    ea2_enemy3 = pickle.load(fp)
with open ('mean_gains_ea3_enemy1', 'rb') as fp:
    ea3_enemy1 = pickle.load(fp)
with open ('mean_gains_ea3_enemy2', 'rb') as fp:
    ea3_enemy2 = pickle.load(fp)

stats.probplot(ea1_enemy1, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea1_enemy2, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea1_enemy3, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea2_enemy1, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea2_enemy2, dist="norm", plot=pylab)
pylab.show()
stats.probplot(ea2_enemy3, dist="norm", plot=pylab)
pylab.show()

print("Enemy 1")
print(scipy.stats.ks_2samp(ea1_enemy1,ea2_enemy1))
print("Enemy 2")
print(scipy.stats.ks_2samp(ea1_enemy2,ea2_enemy2))
print("Enemy 3")
print(scipy.stats.ks_2samp(ea1_enemy3,ea2_enemy3))

print("Enemy 1")
print(scipy.stats.ttest_ind(ea1_enemy1,ea2_enemy1))
print("Enemy 2")
print(scipy.stats.ttest_ind(ea1_enemy2,ea2_enemy2))
print("Enemy 3")
print(scipy.stats.ttest_ind(ea1_enemy3,ea2_enemy3))