# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 20:09:54 2022

@author: 山抹微云
"""

# In[]
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

# In[] CEC 1997

metaData = pd.read_csv('result/comparison_table/meta_detail.csv')
traData = pd.read_csv('result/comparison_table/traditional.csv')
esoData = pd.read_csv('result/comparison_table/eso.csv')

# In[]

eso_mean = []
eso_std  = []
pso_mean = []
pso_std  = []
ga_mean = []
ga_std  = []
de_mean = []
de_std  = []
gwo_mean = []
gwo_std  = []
hho_mean = []
hho_std  = []

for f in range(1,8):
    f_i = f
    gap = 7*5
    f_i-=1
    
    gwo = []
    hho = []
    
    
    for i in range(1, 5):
        gwo.append(metaData['Iter500'].values[f_i+0*gap+i])
        hho.append(metaData['Iter500'].values[f_i+1*gap+i])
    
    gwo = np.array(gwo)
    hho = np.array(hho)
    
    
    de = np.array([])
    ga = np.array([])
    pso = np.array([])
    eso = np.array([])
    
    for j in range(5):
        de  = np.append(de, traData['de_F{}_{}'.format(f_i+1,j)].values[-1])
        ga  = np.append(ga, traData['ga_F{}_{}'.format(f_i+1,j)].values[-1])
        pso = np.append(pso, traData['pso_F{}_{}'.format(f_i+1,j)].values[-1])
        eso = np.append(eso, esoData['eso_F{}_{}'.format(f_i+1,j)].values[-1])

    gwo_mean.append(gwo.mean())
    hho_mean.append(hho.mean())
    de_mean.append(de.mean())
    ga_mean.append(ga.mean())
    pso_mean.append(pso.mean())
    eso_mean.append(eso.mean())
    
    gwo_std.append(gwo.std())
    hho_std.append(hho.std())
    de_std.append(de.std())
    ga_std.append(ga.std())
    pso_std.append(pso.std())
    eso_std.append(eso.std())
    
# In[] CEC 1997 data
data = pd.DataFrame({
        'eso_mean': eso_mean,
        'eso_std' : eso_std,
        'pso_mean': pso_mean,
        'pso_std' : pso_std,
        'ga_mean' : ga_mean,
        'ga_std'  : ga_std,
        'de_mean' : de_mean,
        'de_std'  : de_std,
        'gwo_mean': gwo_mean,
        'gwo_std' : gwo_std,
        'hho_mean': hho_mean,
        'hho_std' : hho_std
    })
    
data.to_csv('result/comparison_table/CEC1997_F1-F7_data.csv')
    
    
# In[] CEC 2017

metaData = pd.read_csv('result/comparison_table/meta_cec17_detail.csv')
traData = pd.read_csv('result/comparison_table/traditional_cec17.csv')

# In[]

eso_mean = []
eso_std  = []
pso_mean = []
pso_std  = []
ga_mean = []
ga_std  = []
de_mean = []
de_std  = []
gwo_mean = []
gwo_std  = []
hho_mean = []
hho_std  = []

for f in range(1,30):
    f_i = f
    gap = 29*5
    f_i-=1
    
    gwo = []
    hho = []
    
    
    for i in range(5):
        
        print(f_i, i, f_i*5+gap+i)
        gwo.append(metaData['Iter500'].values[f_i*5+i])
        hho.append(metaData['Iter500'].values[f_i*5+gap+i])
    
    gwo = np.array(gwo)
    hho = np.array(hho)
    
    
    de = np.array([])
    ga = np.array([])
    pso = np.array([])
    eso = np.array([])
    
    for j in range(5):
        try:
            de  = np.append(de, traData['de_F{}_{}'.format(f_i+1,j)].values[-1])
            ga  = np.append(ga, traData['ga_F{}_{}'.format(f_i+1,j)].values[-1])
            pso = np.append(pso, traData['pso_F{}_{}'.format(f_i+1,j)].values[-1])
            eso = np.append(eso, traData['eso_F{}_{}'.format(f_i+1,j)].values[-1])
        except:
            de  = np.append(de,  np.inf)
            ga  = np.append(ga,  np.inf)
            pso = np.append(pso, np.inf)
            eso = np.append(eso, np.inf)

    gwo_mean.append(gwo.mean())
    hho_mean.append(hho.mean())
    de_mean.append(de.mean())
    ga_mean.append(ga.mean())
    pso_mean.append(pso.mean())
    eso_mean.append(eso.mean())
    
    gwo_std.append(gwo.std())
    hho_std.append(hho.std())
    de_std.append(de.std())
    ga_std.append(ga.std())
    pso_std.append(pso.std())
    eso_std.append(eso.std())
    
# In[] CEC 1997 data
data = pd.DataFrame({
        'eso_mean': eso_mean,
        'eso_std' : eso_std,
        'pso_mean': pso_mean,
        'pso_std' : pso_std,
        'ga_mean' : ga_mean,
        'ga_std'  : ga_std,
        'de_mean' : de_mean,
        'de_std'  : de_std,
        'gwo_mean': gwo_mean,
        'gwo_std' : gwo_std,
        'hho_mean': hho_mean,
        'hho_std' : hho_std
    })

    
data.to_csv('result/comparison_table/CEC2017_data.csv')
    
    
    
    
    

