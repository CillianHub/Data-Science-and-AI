# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:39:28 2021

@author: Cillian
"""

import numpy as np
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
cf.go_offline()

pio.renderers.default='browser'
#pio.renderers.default='svg'  
init_notebook_mode(connected=(True) )
df = pd.read_csv('2014_World_GDP')

data = dict(type='choropleth',
            colorscale = 'Reds',
            locations = df['CODE'],
            z = df['GDP (BILLIONS)'],
            text = df['COUNTRY'],
            marker = dict(line = dict(color = 'rgb(5,5,5)',width = 1)),
            colorbar = {'title':"GDP in Billions USD"}
            ) 

layout = dict(title = '2014_World_GDP',
              geo = dict(showframe = False,
                         projection = {'type': 'stereographic'}
                         )
             )

choromap = go.Figure(data = [data],layout = layout)
choromap.update_geos(
    showocean=True, oceancolor="LightBlue",
    showlakes=True, lakecolor="LightBlue",
    showrivers=True, rivercolor="LightBlue"
)
plot(choromap)
