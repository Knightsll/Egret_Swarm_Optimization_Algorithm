# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 15:10:58 2022

@author: 山抹微云
"""
import numpy as np
from testFunc import *
class BAS:
    def __init__(self, n, step, func, lb, ub):
        self.n = n
        self.lb = lb
        self.ub = ub
        self.x_init()
        self.theta = np.random.random(self.n)-0.5*np.ones(self.n)
        self.dlr = self.theta/np.sqrt(np.sum(self.theta**2))
        self.xr = self.x + self.dlr/2
        self.xl = self.x - self.dlr/2
        self.step = step
        self.backup = step
        self.eta = 0.95
        self.func = func
        
    def checkBound(self, x):
        return np.clip(x, self.lb, self.ub)
    
    def direction_update(self):
        self.theta = np.random.random(self.n)-0.5*np.ones(self.n)
        self.dlr = self.theta/np.sqrt(np.sum(self.theta**2))
        self.xr = self.x + self.dlr/2
        self.xr = self.checkBound(self.xr)
        self.xl = self.x - self.dlr/2
        self.xl = self.checkBound(self.xl)
    def fit(self, g):
        self.step = self.backup
        
        self.f = self.func(self.x)
        f_sum = np.array(self.f)
        for i in range(1,g+1):
            if i%100 == 0:
                self.step=2
            self.f_xl = self.func(self.xl)
            self.f_xr = self.func(self.xr)
            
            if self.f <= min(self.f_xl, self.f_xr):
                f_sum = np.append(f_sum, self.f)
            else:
                if self.f_xl < self.f_xr:
                    self.x-=self.step*self.dlr
                    f_sum = np.append(f_sum, self.f_xl)
                    self.f = self.f_xl
                else:
                    self.x+=self.step*self.dlr
                    f_sum = np.append(f_sum, self.f_xr)
                    self.f = self.f_xr
            self.direction_update()
            self.step*=self.eta

        return f_sum,self.f
    def x_init(self):
        self.x = np.random.uniform(0, 1, size=self.n)* (self.ub-self.lb) + self.lb
        
        
# In[]
        
bas = BAS(30, 100, F1, -100, 100)
bas_hist, bas_best = bas.fit(500*50)








