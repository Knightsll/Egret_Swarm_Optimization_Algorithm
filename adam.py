# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 00:15:19 2022

@author: 山抹微云
"""

# In[]
import numpy as np
import pandas as pd
import seaborn as sb
from testFunc import Sphere, Schwefel_2_22, Rosenbrock

# In[]

n_dim = 30
size_pop = 50
max_iter = 800
lb   = np.array([-30]*30)
ub   = np.array([30]*30)

# In[]

beta1 = 0.9
beta2 = 0.99

adam_x = np.random.uniform(lb, ub, size=(size_pop, n_dim))
adam_g = np.empty_like(adam_x)
adam_m = np.zeros((size_pop, n_dim))
adam_v = np.zeros((size_pop, n_dim))
adam_y = np.empty(size_pop)
last_y = np.empty(size_pop)
adam_y_hist = []
for times in range(200):
    
    for i in range(size_pop):
        adam_y[i] = Rosenbrock(adam_x[i,:])
        adam_g[i,:] = (adam_y[i]- last_y[i])*adam_x[i,:]

        adam_m[i,:] = beta1*adam_m[i,:]+(1-beta1)*adam_g[i,:]
        adam_v[i,:] = beta2*adam_v[i,:]+(1-beta2)*adam_g[i,:]**2
        adam_x[i,:] = adam_x[i,:] - 10*adam_m[i,:]/(np.sqrt(adam_v[i,:])+np.spacing(1))
    last_y = adam_y.copy()

    adam_y_hist.append(adam_y.min())












