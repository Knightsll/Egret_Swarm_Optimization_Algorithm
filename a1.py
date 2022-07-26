# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 19:07:41 2022

@author: 山抹微云
"""


# In[]
import numpy as np
import pandas as pd
import seaborn as sb
from testFunc import Sphere, Schwefel_2_22, Rosenbrock, Schwefel_2_26
from scipy.special import gamma
import math

from cec2017.functions import *

# In[]

func     = f5
n_dim = 30
size_pop = 50
max_iter = 1000
lb   = np.array([-100]*n_dim)
ub   = np.array([100]*n_dim)
# In[]

import time

s = time.time()

beta1 = 0.9
beta2 = 0.99


x = np.random.uniform(lb, ub, size=(size_pop, n_dim))
w = np.random.uniform(lb, ub, size=(size_pop, n_dim))
g = np.empty_like(x)
m = np.zeros((size_pop, n_dim))
v = np.zeros((size_pop, n_dim))
y = np.empty(size_pop)

p_y = y.copy()

y_hist = []
err = []

x_hist_best = np.zeros_like(x)
g_hist_best = np.zeros_like(x)
y_hist_best = np.ones(size_pop)*np.inf

x_global_best = np.zeros(n_dim)
g_global_best = np.zeros(n_dim)
y_global_best = func(x[0])

w_hist = []

hop = ub - lb

beta = 2/3
alpha_u = math.pow((gamma(1+beta)*math.sin(math.pi*beta/2)/(gamma( ((1+beta)/2)*beta*math.pow(2,(beta-1)/2)) ) ),(1/beta))
alpha_v = 1


def checkBound(x):
    try:
        r, c = x.shape
        for i in range(r):
            for j in range(c):
                x[i][j] = lb[j] if x[i][j] < lb[j] else x[i][j]
                x[i][j] = ub[j] if x[i][j] > ub[j] else x[i][j]
    except:
        r, = x.shape
        for i in range(r):
            x[i] = lb[i] if x[i] < lb[i] else x[i]
            x[i] = ub[i] if x[i] > ub[i] else x[i]
    return x


choice = []
T = 100

x_c = x[0,:]
y_c = func(x_c)

best_x = np.random.uniform(lb, ub, size=n_dim)
best_y = func(best_x)

x_current, y_current = best_x, best_y
stay_counter = 0

def get_new_x(x):
    r = np.random.uniform(-np.pi / 2, np.pi / 2, size=n_dim)
    xc = 0.5 * T * np.tan(r)
    x_new = x + xc

    return np.clip(x_new, lb, ub)

def cool_down(iter_cycle):
    T = 100 * (1+iter_cycle)
    return T


for k in range(max_iter):
    for i in range(size_pop):
        x_new = get_new_x(x_current)
        y_new = func(x_new)

        # Metropolis
        df = y_new - y_current
        if df < 0 or np.exp(-df / T) > np.random.rand():
            x_current, y_current = x_new, y_new
            if y_new < best_y:
                best_x, best_y = x_new, y_new
        


    cool_down(k)
    y_hist.append(best_y)



    if T < 10e-6:
        stop_code = 'Cooled to final temperature'
        print(stop_code)
        break
e = time.time()
print(e-s)







