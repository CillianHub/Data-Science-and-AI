# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:26:26 2021

@author: Cillian
"""

import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')
iris['species'].unique()
g = sns.PairGrid(iris)  
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

 g = sns.FacetGrid(data = tips, col = 'time', row = 'smoker')
 g.map(sns.distplot, 'total_bill', kde = False)
g.map(plt.scatter, 'total_bill', 'tip')

sns.lmplot(x = 'total_bill', y = 'tip', data = tips, hue = 'sex', markers = ['o','v'],scatter_kws= {'s':50})

sns.lmplot(x = 'total_bill', y = 'tip', data = tips, col = 'sex', row = 'time')

sns.lmplot(x = 'total_bill', y = 'tip', data = tips, col = 'day', hue = 'sex', aspect= 0.6, size = 8)
