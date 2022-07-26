# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:31:58 2022

@author: 山抹微云
"""
# In[]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

# In[]

metaData = pd.read_csv('result/convergence/meta.csv')
traData = pd.read_csv('result/convergence/traditional.csv')
esoData = pd.read_csv('result/convergence/eso.csv')
# In[]
f_i = 6
gap = 7
f_i-=1

gwo = []
hho = []


for i in range(1, 501):
    gwo.append(metaData['Iter{}'.format(i)].values[f_i+0*gap])
    hho.append(metaData['Iter{}'.format(i)].values[f_i+1*gap])





de = np.zeros(500)
ga = np.zeros(500)
pso = np.zeros(500)
eso = np.zeros(500)

for j in range(5):
    de = de + traData['de_F{}_{}'.format(f_i+1,j)].values
    ga = ga + traData['ga_F{}_{}'.format(f_i+1,j)].values
    pso = pso + traData['pso_F{}_{}'.format(f_i+1,j)].values
    eso = eso + esoData['eso_F{}_{}'.format(f_i+1,j)].values

de/=5
ga/=5
pso/=5
eso/=5





comparisons = pd.DataFrame({
        'GWO' : gwo,
        'HHO':hho,
        'PSO':pso,
        'GA':ga,
        'DE':de,
        'ESOA': eso
        
    })
fig = plt.figure(figsize=(12,9))

sb.set_theme(style="ticks")
ax = fig.gca()

ax.plot(np.arange(500), eso, color='#F75000', alpha=0.8, label='ESOA', linestyle='--')
ax.plot(np.arange(500), gwo, color='#8600FF', alpha=0.8, label='GWO',  linestyle='--')
ax.plot(np.arange(500), hho, color='#2894FF', alpha=0.8, label='HHO',  linestyle='--')
ax.plot(np.arange(500), pso, color='#00CACA', alpha=0.8, label='PSO',  linestyle='--')
ax.plot(np.arange(500), ga,  color='#8080C0', alpha=0.8, label='GA',   linestyle='--')
ax.plot(np.arange(500), de,  color='#D2A2CC', alpha=0.8, label='DE',   linestyle='--')
ax.legend(loc="upper right", fontsize=24)
ax.set_xlabel('Iteration', fontsize=26)
ax.set_ylabel('Fitness', fontsize=26)

ax.set_ylim(-10, 120)

axins = inset_axes(ax, width="35%", height="35%", loc='center left',
                   bbox_to_anchor=(0.3, 0.1, 1, 1), 
                   bbox_transform=ax.transAxes)

axins.plot(np.arange(500), eso, color='#F75000', alpha=0.8, label='ESOA', linestyle='--')
axins.plot(np.arange(500), gwo, color='#8600FF', alpha=0.8, label='GWO',  linestyle='--')
axins.plot(np.arange(500), hho, color='#2894FF', alpha=0.8, label='HHO',  linestyle='--')
axins.plot(np.arange(500), pso, color='#00CACA', alpha=0.8, label='PSO',  linestyle='--')
axins.plot(np.arange(500), ga,  color='#8080C0', alpha=0.8, label='GA',   linestyle='--')
axins.plot(np.arange(500), de,  color='#D2A2CC', alpha=0.8, label='DE',   linestyle='--')

xlim0 = -1
xlim1 = 50
ylim0 = -1
ylim1 = 4
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)


tx0 = xlim0
tx1 = xlim1
ty0 = ylim0*1
ty1 = ylim1*1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

xy = (xlim0,ty1)
xy2 = (xlim0,ylim0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax, color='black', linestyle=':')
axins.add_artist(con)

xy = (xlim1,ty1)
xy2 = (xlim1,ylim0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax, color='black', linestyle=':')
axins.add_artist(con)



# In[]


# In[]

metaData = pd.read_csv('result/convergence/meta_cec17_all.csv')
traData = pd.read_csv('result/convergence/traditional_cec17.csv')

# In[]
f_i = 26 
gap = 29
f_i-=1

gwo = []
hho = []

ylim_l, ylim_u = 2900, 18000

xlim0, xlim1 = 50, 150
ylim0, ylim1 = 5000, 7000
 

for i in range(1, 501):
    gwo.append(metaData['Iter{}'.format(i)].values[f_i+0*gap])
    hho.append(metaData['Iter{}'.format(i)].values[f_i+1*gap])





de = np.zeros(500)
ga = np.zeros(500)
pso = np.zeros(500)
eso = np.zeros(500)

for j in range(5):
    de = de + traData['de_F{}_{}'.format(f_i+1,j)].values
    ga = ga + traData['ga_F{}_{}'.format(f_i+1,j)].values
    pso = pso + traData['pso_F{}_{}'.format(f_i+1,j)].values
    eso = eso + traData['eso_F{}_{}'.format(f_i+1,j)].values

de/=5
ga/=5
pso/=5
eso/=5



comparisons = pd.DataFrame({
        'GWO' : gwo,
        'HHO':hho,
        'PSO':pso,
        'GA':ga,
        'DE':de,
        'ESOA': eso
        
    })
fig = plt.figure(figsize=(12,9))

sb.set_theme(style="ticks")
ax = fig.gca()

ax.plot(np.arange(500), eso, color='#F75000', alpha=0.8, label='ESOA', linestyle='--')
ax.plot(np.arange(500), gwo, color='#8600FF', alpha=0.8, label='GWO',  linestyle='--')
ax.plot(np.arange(500), hho, color='#2894FF', alpha=0.8, label='HHO',  linestyle='--')
ax.plot(np.arange(500), pso, color='#00CACA', alpha=0.8, label='PSO',  linestyle='--')
ax.plot(np.arange(500), ga,  color='#8080C0', alpha=0.8, label='GA',   linestyle='--')
ax.plot(np.arange(500), de,  color='#D2A2CC', alpha=0.8, label='DE',   linestyle='--')
ax.legend(loc="upper right", fontsize=24)
ax.set_xlabel('Iteration', fontsize=26)
ax.set_ylabel('Fitness', fontsize=26)

ax.set_ylim(ylim_l, ylim_u)

axins = inset_axes(ax, width="35%", height="35%", loc='center left',
                   bbox_to_anchor=(0.3, 0.1, 1, 1), 
                   bbox_transform=ax.transAxes)

axins.plot(np.arange(500), eso, color='#F75000', alpha=0.8, label='ESOA', linestyle='--')
axins.plot(np.arange(500), gwo, color='#8600FF', alpha=0.8, label='GWO',  linestyle='--')
axins.plot(np.arange(500), hho, color='#2894FF', alpha=0.8, label='HHO',  linestyle='--')
axins.plot(np.arange(500), pso, color='#00CACA', alpha=0.8, label='PSO',  linestyle='--')
axins.plot(np.arange(500), ga,  color='#8080C0', alpha=0.8, label='GA',   linestyle='--')
axins.plot(np.arange(500), de,  color='#D2A2CC', alpha=0.8, label='DE',   linestyle='--')


axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)


tx0 = xlim0
tx1 = xlim1
ty0 = ylim0*1
ty1 = ylim1*1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

xy = (xlim0,ty1)
xy2 = (xlim0,ylim0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax, color='black', linestyle=':')
axins.add_artist(con)

xy = (xlim1,ty1)
xy2 = (xlim1,ylim0)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax, color='black', linestyle=':')
axins.add_artist(con)

















