# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:35:31 2022

@author: 山抹微云
"""
# In[]
import numpy as np



# In[]
class ESO:
    def __init__(self, func, n_dim, size_pop, max_iter, lb, ub, constraint_ueq=None):
        
        self.func     = func
        
        self.n_dim    = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.lb   = lb
        self.ub   = ub
        
        # adam's learning rate of weight estimate
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.m = np.zeros((self.size_pop, self.n_dim))
        self.v = np.zeros((self.size_pop, self.n_dim))
        self.w = np.random.uniform(-1, 1, size=(self.size_pop, self.n_dim))
        self.g = np.empty_like(self.w)
        
        # location, fitness, and estimate fitness
        self.x = np.random.uniform(0, 1, size=(self.size_pop, self.n_dim)) * (self.ub-self.lb) + self.lb
        self.y = np.empty(self.size_pop)
        self.p_y = self.y.copy()
        
        # best fitness history and estimate error history
        self.y_hist = []
        self.err = []
        
        # individual best location, gradient direction, and fitness 
        self.x_hist_best = self.x.copy()
        self.g_hist_best = np.empty_like(self.x)
        self.y_hist_best = np.ones(size_pop)*np.inf
        
        # group best location, gradient direction, and fitness 
        self.x_global_best = self.x[0].copy()
        self.g_global_best = np.zeros(self.n_dim)
        self.y_global_best = func(self.x[0])
        
        self.hop = self.ub - self.lb
        
    
    def checkBound(self,x):
        return np.clip(x, self.lb, self.ub)
    
    

    def refill(self,V):
        V = V.reshape(len(V), 1)
        V = np.tile(V, self.n_dim)
        return V
    
    def gradientEstimate(self, g_temp):
        
        # Indivual direction
        p_d = self.x_hist_best - self.x
        p_d_sum = p_d.sum(axis=1)
        p_d_sum = self.refill(p_d_sum)
        f_p_bias = self.y_hist_best - self.y
        f_p_bias = self.refill(f_p_bias)
        p_d *= f_p_bias
        p_d /= (p_d_sum+np.spacing(1))*(p_d_sum+np.spacing(1))
        
        d_p = p_d + self.g_hist_best
        
        # Group direction
        c_d = self.x_global_best - self.x
        c_d_sum = c_d.sum(axis=1)
        c_d_sum = self.refill(c_d_sum)
        f_c_bias = self.y_global_best - self.y
        f_c_bias = self.refill(f_c_bias)
        c_d *= f_c_bias
        c_d /= (c_d_sum+np.spacing(1))*(p_d_sum+np.spacing(1))
        
        d_g = c_d + self.g_global_best
        
        # Advice
        r1 = np.random.random(self.size_pop)
        r1 = self.refill(r1)
        
        r2 = np.random.random(self.size_pop)
        r2 = self.refill(r2)
        
        r3 = np.random.random(self.size_pop)
        r3 = self.refill(r3)
        
        self.g = r1 * g_temp + r2 * d_p + r3 * d_g
        g_sum = self.g.sum(axis=1)
        g_sum = self.refill(g_sum)
        self.g /= (g_sum+np.spacing(1))
    
    def weightUpdate(self):
        # Update weight
        self.m = self.beta1*self.m+(1-self.beta1)*self.g
        self.v = self.beta2*self.v+(1-self.beta2)*self.g**2
        self.w = self.w - self.m/(np.sqrt(self.v)+np.spacing(1))
    
    def updateSurface(self):
        self.y = np.array([self.func(self.x[i, :]) for i in range(self.size_pop)]).reshape(self.size_pop)
        self.p_y = np.sum(self.w*self.x, axis=1)
        self.err.append(np.abs(self.y-self.p_y).min())
        p = self.p_y-self.y
        p = self.refill(p)
        g_temp = p*self.x
        
        mask = self.y < self.y_hist_best
        self.y_hist_best = np.where(mask, self.y, self.y_hist_best)
        
        mask = self.refill(mask)
        self.x_hist_best = np.where(mask, self.x, self.x_hist_best)
        self.g_hist_best = np.where(mask, g_temp, self.g_hist_best)
        
        g_hist_sum = self.refill(np.sqrt((self.g_hist_best**2).sum(axis=1)))
        self.g_hist_best /= (g_hist_sum + np.spacing(1))
        
        # Data Insecure
        if self.y.min() < self.y_global_best:
            self.y_global_best = self.y.min()
            self.x_global_best = self.x[self.y.argmin(), :]
            self.g_global_best = g_temp[self.y.argmin(), :]
            self.g_global_best /= np.sqrt(np.sum(self.g_global_best**2)) 
        
        self.gradientEstimate(g_temp)
        
        self.weightUpdate()
        
    

    def randomSearch(self):
        # Random search
        r = np.random.uniform(-np.pi / 2, np.pi / 2, size=(self.size_pop, self.n_dim))
        x_n = self.x + np.tan(r) * self.hop/( 1 + self.times) *0.5
        
        x_n = self.checkBound(x_n)

        
        y_n = np.array([self.func(x_n[i, :]) for i in range(self.size_pop)])
        
        # Random step search
        d = self.x_hist_best - self.x
        d_g = self.x_global_best - self.x
        #r = np.random.uniform(-np.pi / 2, np.pi / 2, size=(size_pop, n_dim))
        r = np.random.uniform(0, 0.5, size=(self.size_pop, self.n_dim))
        r2 = np.random.uniform(0, 0.5, size=(self.size_pop, self.n_dim))
        x_m = (1-r-r2) * self.x + r * d + r2 * d_g
        x_m = self.checkBound(x_m)

        
        y_m = np.array([self.func(x_m[i, :]) for i in range(self.size_pop)])

        return x_m, y_m, x_n, y_n


    def adviceSearch(self):
        x_o = self.x + np.exp(-self.times/(0.1*self.max_iter)) * 0.1 * self.hop * self.g
        x_o = self.checkBound(x_o)

        
        y_o = np.array([self.func(x_o[i, :]) for i in range(self.size_pop)])
        return x_o, y_o

    
    def run(self):
        for self.times in range(self.max_iter):

            self.updateSurface()
            x_m, y_m, x_n, y_n = self.randomSearch()
            x_o, y_o = self.adviceSearch()
            
            # Comparison
            x_i = np.empty_like(self.x)
            y_i = np.empty_like(self.y)
            x_summary = np.array([x_m, x_n, x_o])
            y_summary = np.column_stack((y_m, y_n, y_o))
            y_summary[y_summary == np.nan] = np.inf
            i_ind = y_summary.argmin(axis=1)
            for i in range(self.size_pop):
                y_i[i] = y_summary[i, i_ind[i]]
                x_i[i,:] = x_summary[i_ind[i]][i]

            
            # Update location
            
            mask = y_i < self.y
            self.y = np.where(mask, y_i, self.y)
            
            mask = self.refill(mask)
            self.x = np.where(mask, x_i, self.x)
            
            
            mask = y_i < self.y_hist_best
            self.y_hist_best = np.where(mask, y_i, self.y_hist_best)
            
            mask = self.refill(mask)
            self.x_hist_best = np.where(mask, x_i, self.x_hist_best)
            
            if y_i.min() < self.y_global_best:
                self.y_global_best = y_i[y_i.argmin()]
                self.x_global_best = x_i[y_i.argmin(), :]
            else:
                ran = np.random.random(self.size_pop)
                ran = self.refill(ran)
                ran[mask] = 1
                mask = ran < 0.3
                self.x = np.where(mask, x_i, self.x)
                self.y = np.where(mask[:, 0], y_i, self.y)
            

            

            self.y_hist.append(self.y_global_best.copy())


