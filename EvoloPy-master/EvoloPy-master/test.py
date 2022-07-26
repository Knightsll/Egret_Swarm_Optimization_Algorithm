# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 02:11:01 2022

@author: 山抹微云
"""

# In[]
import pandas as pd
import seaborn as sb

# In[]

data = pd.read_csv('2022-07-09-02-16-47/experiment.csv')
# In[]
ssa = []
bat = []
gwo = []
mvo = []
jaya = []
for i in range(1, 201):
    ssa.append(data['Iter{}'.format(i)].values[0])
    bat.append(data['Iter{}'.format(i)].values[1])
    gwo.append(data['Iter{}'.format(i)].values[2])
    mvo.append(data['Iter{}'.format(i)].values[3])
    jaya.append(data['Iter{}'.format(i)].values[4])









