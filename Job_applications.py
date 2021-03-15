# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:06:04 2021

@author: Cillian
"""

import numpy as np
import pandas as pd

d = {'Company':['Deloitte'],'Job Title':['Junior Release Engineer'],'Date Applied':['10th March']}
df = pd.DataFrame(d)

df['Interviewed'] = 'no'
df.loc[1] = ['CPL']+['Engineer All Levels'] + ['10th March'] +['no']
df.loc[2] = ['Propylon']+['Junior Software Engineer'] + ['10th March'] + ['no']
df.loc[3] = ['Hays Group']+['Applications Engineer'] + ['10th March'] + ['no']
df.loc[4] = ['PM Group']+['Engineering Oppertunities'] + ['10th March'] + ['no']
df.loc[5] = ['Intercom']+['Software Engineer'] + ['10th March'] + ['no']
df.loc[6] = ['Workday']+['Software Application Engineer'] + ['10th March'] + ['no']
df.loc[7] = ['Workday']+['Software Dev Engineer Analytics'] + ['10th March'] + ['no']
df.loc[8] = ['Huawei']+['Data Scientist-Smart Networks'] + ['10th March'] + ['no']
df.loc[9] = ['Huawei']+['AI Researcher - Intern'] + ['10th March'] + ['no']
df.loc[10] = ['Huawei']+['Researcher - Open Source Software'] + ['10th March'] + ['no']
df.loc[11] = ['Riot Games']+['Systems Engineer'] + ['11th March'] + ['no']
df.loc[12] = ['SIG']+['Grad Software Engineer'] + ['11th March'] + ['no']
df.loc[13] = ['Intercom']+['Data Scientist'] + ['11th March'] + ['no']
df.loc[14] = ['Kitman Labs']+['Data Scientist'] + ['11th March'] + ['no']
df.loc[15] = ['Intercom']+['Software Engineer'] + ['11th March'] + ['no']
df.loc[16] = ['Vectra AI']+['Associate Analyst'] + ['11th March'] + ['no']
df.loc[17] = ['Patreon']+['Data Scientist'] + ['11th March'] + ['no']
df.loc[18] = ['Reperio']+['Data Scientist'] + ['11th March'] + ['no']
df.loc[19] = ['Intel']+['Product Dev Engineer'] + ['11th March'] + ['no']
df['Refused'] = 'no'
df.loc[df.Company == 'PM Group', 'Refused']  = 'yes'
df.loc[20] = ['Sabeo']+['Data Engineer'] + ['12th March'] + ['no'] + ['no']
df.loc[21] = ['2K']+['Data Engineer(Video Games)'] + ['12th March'] + ['no'] + ['no']
df.loc[22] = ['Canonical']+['Software Engineer(Micro K8s)'] + ['12th March'] + ['no'] + ['no']
df.loc[23] = ['Intuition']+['Data Engineer'] + ['12th March'] + ['no'] + ['no']
df.loc[24] = ['Optum']+['Data Scientist'] + ['12th March'] + ['no'] + ['no']
df.loc[25] = ['Quantum']+['Build Eningeer'] + ['12th March'] + ['no'] + ['no']
df.loc[26] = ['Datadog']+['Solutions Eningeer'] + ['12th March'] + ['no'] + ['no']
df.loc[27] = ['Dunnhumby']+['Applied Data Scientist'] + ['12th March'] + ['no'] + ['no']
df.loc[28] = ['Intel']+['Business Analytics Engineer'] + ['12th March'] + ['no'] + ['no']
df.loc[29] = ['Shutterstock']+['Data Engineer - Computer Vision'] + ['15th March'] + ['no'] + ['no']
df.loc[30] = ['CreVinn Teoranta']+['ASIC / FPGA Digital Electronics Design'] + ['15th March'] + ['no'] + ['no']
df['Website'] = 'LinkedIn'
df.loc[31] = ['Astatine']+['Project Engineer'] + ['15th March'] + ['no'] + ['no'] +['indeed']
df.loc[df.Company == 'CreVinn Teoranta', 'Website']  = 'indeed'  
df.loc[32] = ['Alldus']+['Data Scientist'] + ['15th March'] + ['no'] + ['no'] +['indeed']
df.loc[33] = ['IT Search']+['Data Scientist'] + ['15th March'] + ['no'] + ['no'] +['indeed']
df.loc[34] = ['Cubic Telecom']+['Junior Telecoms Engineer'] + ['15th March'] + ['no'] + ['no'] +['indeed']
df.loc[35] = ['Sage']+['Data Scientist'] + ['15th March'] + ['no'] + ['no'] +['indeed']





df.loc[df.Company == 'Patreon', 'Refused']  = 'yes'
df.loc[df.Company == 'Workday', 'Refused']  = 'yes'
df.loc[df.Company == 'Canonical', 'Refused']  = 'yes'


df[df['Company']=='Microchip']



#to replace a defined value in a defined column
df.loc[df.Company == 'Astatine', 'Website']  = 'indeed'  

#to find a defined value in a defined column
test_df = df[df['Company'] == 'Deloitte']['Date Applied']

]

return_company = df[df['Job Title'].str.contains('Data')]
return_company = df[df['Company']== 'Huawei']