# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:07:55 2021

@author: adehg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Developed by adehghanGitHub in February 2021.
# This is a work in progress. 
# The code comapres the financial development indicators of different countries.
# The data is from the "Global Financial Development Database".
#data link at https://www.worldbank.org/en/publication/gfdr/data/global-financial-development-database

df=pd.read_excel('https://www.worldbank.org/en/publication/gfdr/data/global-financial-development-database/October2019globalfinancialdevelopmentdatabase.xlsx')
#list of countries of ineterst for comparison purposes.
country_list=['Canada','Germany','Japan','United States']

plt.figure(figsize=(8,8))
ax=plt.subplot() 
plt.tight_layout(pad=6.0)
for i in country_list:
    df_country=df[df['country']==i]
    df_country.set_index('year',inplace=True)
#plot time series of finacial development indicators for countries of interest
    df_country.plot(y='gfdddi14', use_index=True,ax=ax,title='Annual', fontsize=15)    
    ax.legend(country_list)
    ax.set_xlabel("Year", fontsize=15)
    ax.set_ylabel("Credit to Private Sector", fontsize=15)