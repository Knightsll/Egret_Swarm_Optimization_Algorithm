# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:13:15 2022

@author: 山抹微云
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from testFunc import Sphere, Rosenbrock

class PSO:
    def __init__(self, dimension, maxIter, size, xmin, xmax, v_min, v_max, func):
        self.func   = func
        self.dimension = dimension
        self.maxIter   = maxIter
        self.size      = size
        self.xmin      = xmin
        self.xmax      = xmax
        self.v_min     = v_min
        self.v_max     = v_max
        self.pso_x = np.random.random((self.size, self.dimension))*(xmax-xmin)+xmin
        self.pso_v = np.random.random((self.size, self.dimension))*(v_max-v_min)+v_min

        self.p_best = self.pso_x.copy()
        self.p_best_fit = np.array([self.func(self.pso_x[i, :]) for i in range(self.size)])
        
        self.g_best = self.p_best[np.where(self.p_best_fit==self.p_best_fit.min())[0][0], :]
        self.g_best_fit = self.p_best_fit[np.where(self.p_best_fit==self.p_best_fit.min())[0][0]]

    def update(self):
        c1 = 2.0  
        c2 = 2.0
        w = 0.8  
        for i in range(self.size):
            self.pso_v[i, :] = w * self.pso_v[i, :] + c1 * np.random.random() * (
                    self.p_best[i, :] - self.pso_x[i, :]) + c2 * np.random.random() * (self.g_best - self.pso_x[i, :])
            
            for j in range(self.dimension):
                
                if self.pso_v[i][j] < self.v_min:
                    self.pso_v[i][j] = self.v_min
                if self.pso_v[i][j] > self.v_max:
                    self.pso_v[i][j] = self.v_max

            
            self.pso_x[i, :] = self.pso_x[i, :] + self.pso_v[i, :]
            
            for j in range(self.dimension):
                
                if self.pso_x[i][j] < self.xmin:
                    self.pso_x[i][j] = self.xmin
                if self.pso_x[i][j] > self.xmax:
                    self.pso_x[i][j] = self.xmax
            
            if self.func(self.pso_x[i, :]) < self.func(self.p_best[i, :]):
                self.p_best[i, :] = self.pso_x[i, :]
                self.p_best_fit[i] = self.func(self.p_best[i, :])
            if self.func(self.pso_x[i, :]) < self.func(self.g_best):
                self.g_best = self.pso_x[i, :]
                self.g_best_fit = self.func(self.g_best)

    def iteration(self):
        self.f_history = []
        self.xbest = self.g_best.copy()
        self.fbest = self.g_best_fit
        for i in range(self.maxIter):
            self.update()
            if self.g_best_fit < self.fbest:
                self.xbest = self.g_best.copy()
                self.fbest = self.g_best_fit
                self.f_history.append(self.g_best_fit)
            else:
                self.f_history.append(self.fbest)
        
                



        
        
        
