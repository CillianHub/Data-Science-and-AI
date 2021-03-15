# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:47:47 2021

@author: Cillian
"""
import numpy as np
import scipy
import seaborn as sns
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')
tips.head(5)
flights.head(3)

sns.distplot(tips['total_bill'], fit = scipy.stats.norm)
sns.jointplot('total_bill','tip',data =  tips, kind = 'hex')
sns.pairplot(tips, hue = 'sex', palette = 'inferno')

sns.boxplot(x = 'day',y = 'total_bill', data = tips, hue = 'smoker')
sns.violinplot(x = 'day', y = 'total_bill', data = tips)
sns.violinplot(x = 'day', y = 'total_bill', data = tips, hue = 'sex', split = True)

sns.stripplot(x = 'day', y = 'total_bill', data = tips, jitter = False)
sns.stripplot(x = 'day', y = 'total_bill', data = tips, hue = 'sex', jitter = True)
sns.swarmplot(x = 'day', y = 'total_bill', data = tips)


sns.violinplot(x = 'day', y = 'total_bill', data = tips)
sns.swarmplot(x = 'day', y = 'total_bill', data = tips, color = 'black')

x = tips.corr()
x
sns.heatmap(x, annot = True)

fp = flights.pivot_table(index = 'month', columns = 'year', values = 'passengers')
sns.heatmap(fp)
sns.heatmap(fp, cmap = 'cool',linecolor='black',linewidths=1) 
sns.clustermap(fp)
