# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:59:17 2022

@author: 山抹微云
"""

from testFunc import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from cec2017.functions import *

from ESO import ESO

# In[]
from sko.DE  import DE
from sko.GA  import GA
from sko.PSO import PSO
data ={}
for i in range(21, 29):
    for j in range(5):
        print(i,j)
        func     = eval('f{}'.format(i))
        n_dim = 30
        size_pop = 50
        max_iter = 500
        lb   = np.array([-100]*n_dim)
        ub   = np.array([100]*n_dim)
        
        
        
        
        de = DE(func=func, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub)
        de.run()
        print('de: ', de.generation_best_Y[-1])
        
        
        ga = GA(func=func, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, prob_mut=0.001, lb=lb, ub=ub, precision=1e-7)
        ga.run()
        print('ga: ', ga.generation_best_Y[-1])
        
        
        pso = PSO(func=func, n_dim=n_dim, pop=size_pop, max_iter=max_iter, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
        pso.run()
        print('pso: ', pso.gbest_y_hist[-1])
        
        eso = ESO(func = func, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub)
        eso.run()
        print('eso: ', eso.y_hist[-1])
        
        data['eso_F{}_{}'.format(i, j)]  = eso.y_hist
        
        data['de_F{}_{}'.format(i, j)]  = de.generation_best_Y
        data['ga_F{}_{}'.format(i, j)]  = ga.generation_best_Y
        data['pso_F{}_{}'.format(i, j)] = pso.gbest_y_hist


# In[]

data = pd.DataFrame(data)

data.to_csv('result/convergence/traditional_f21-f28.csv')
# In[]

data ={}
for i in range(1, 8):
    for j in range(5):
        print(i,j)
        func     = eval('F{}'.format(i))
        n_dim = 30
        size_pop = 50
        max_iter = 500
        lb   = -100
        ub   = 100
        
        
        
        
        eso = ESO(func = func, n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, lb=lb, ub=ub)
        eso.run()
        
        data['eso_F{}_{}'.format(i, j)]  = eso.y_hist

# In[]

data = pd.DataFrame(data)

data.to_csv('result/convergence/eso.csv')







