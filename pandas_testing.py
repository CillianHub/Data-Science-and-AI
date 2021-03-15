# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:40:03 2021

@author: Cillian
"""

import numpy as np
import pandas as  pd
from numpy.random import randn

np.random.seed(101)

df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])

#%%

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df = pd.DataFrame(randn(6,2),hier_index,['A','B'])
df.loc['G1']
df['A']
