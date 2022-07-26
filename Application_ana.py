# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:12:46 2022

@author: 山抹微云
"""


# In[]
import numpy as np
import pandas as pd

# In[]

data = pd.read_csv('result/applications/Himmelblau_solution.csv')
otherData = pd.read_csv('result/applications/Himmelblau_solution_others.csv')
metaData  = pd.read_csv('result/applications/Himmelblau_solution_meta.csv')

eso_solution = []
pso_solution = []
ga_solution  = []
de_solution  = []
gwo_solution = []
hho_solution = []


for i in range(30):
    eso_solution.append(data['Himmelblau_solution_{}'.format(i)].values[-1])
    pso_solution.append(otherData['Himmelblau_solution_pso_{}'.format(i)].values[-1])
    ga_solution.append(otherData['Himmelblau_solution_ga_{}'.format(i)].values[-1])
    de_solution.append(otherData['Himmelblau_solution_de_{}'.format(i)].values[-1])
    
gwo_solution = metaData['Iter500'].values[:30]
hho_solution = metaData['Iter500'].values[30:]

eso_solution = np.array(eso_solution)
pso_solution = np.array(pso_solution)
ga_solution  = np.array(ga_solution)
de_solution  = np.array(de_solution)
gwo_solution = np.array(gwo_solution)
hho_solution = np.array(hho_solution)

print('eso:', eso_solution.min(), eso_solution.max(), eso_solution.mean(), eso_solution.std())
print('pso:', pso_solution.min(), pso_solution.max(), pso_solution.mean(), pso_solution.std())
print('ga:', ga_solution.min(), ga_solution.max(), ga_solution.mean(), ga_solution.std())
print('de:', de_solution.min(), de_solution.max(), de_solution.mean(), de_solution.std())
print('gwo:', gwo_solution.min(), gwo_solution.max(), gwo_solution.mean(), gwo_solution.std())
print('hho:', hho_solution.min(), hho_solution.max(), hho_solution.mean(), hho_solution.std())



# In[]



data = pd.read_csv('result/applications/Spring_solution.csv')
otherData = pd.read_csv('result/applications/Spring_solution_others.csv')
metaData  = pd.read_csv('result/applications/Spring_solution_meta.csv')

eso_solution = []
pso_solution = []
ga_solution  = []
de_solution  = []
gwo_solution = []
hho_solution = []


for i in range(30):
    eso_solution.append(data['spring_solution_{}'.format(i)].values[-1])
    pso_solution.append(otherData['spring_solution_pso_{}'.format(i)].values[-1])
    ga_solution.append(otherData['spring_solution_ga_{}'.format(i)].values[-1])
    de_solution.append(otherData['spring_solution_de_{}'.format(i)].values[-1])
    
gwo_solution = metaData['Iter500'].values[:30]
hho_solution = metaData['Iter500'].values[30:]

eso_solution = np.array(eso_solution)
pso_solution = np.array(pso_solution)
ga_solution  = np.array(ga_solution)
de_solution  = np.array(de_solution)
gwo_solution = np.array(gwo_solution)
hho_solution = np.array(hho_solution)

print('eso:', eso_solution.min(), eso_solution.max(), eso_solution.mean(), eso_solution.std())
print('pso:', pso_solution.min(), pso_solution.max(), pso_solution.mean(), pso_solution.std())
print('ga:', ga_solution.min(), ga_solution.max(), ga_solution.mean(), ga_solution.std())
print('de:', de_solution.min(), de_solution.max(), de_solution.mean(), de_solution.std())
print('gwo:', gwo_solution.min(), gwo_solution.max(), gwo_solution.mean(), gwo_solution.std())
print('hho:', hho_solution.min(), hho_solution.max(), hho_solution.mean(), hho_solution.std())








