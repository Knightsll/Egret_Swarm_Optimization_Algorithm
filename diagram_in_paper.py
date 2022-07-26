# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:41:32 2022

@author: 山抹微云
"""

# In[]


# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from testFunc import *
from scipy.special import gamma
import math

from cec2017.functions import *
import time



from mpl_toolkits import mplot3d

# In[]

def calFitness(X,Y,func):
    z = np.empty_like(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            z[i,j] = func([X[i,j],Y[i,j]])
    return z


x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)

z1 = calFitness(X, Y, f1)


from matplotlib import cm
fig = plt.figure(figsize=(8,8))

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, z1, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none', antialiased=False)
ax.contour(X, Y, z1, zdir='z', offset=2400, cmap=cm.coolwarm)
#ax.set_title('surface')
ax.view_init(20, 30)
plt.show()














