# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:44:36 2020

@author: adehg
"""
#Developed by adehghanGitHub in August 2020.
# This code plots time series of Covid 19 cases, and determines
# the similarity of time series of Covid 19 cases in countries of interest.
# The data is from the "European Centre for Disease Prevention and Control" available at
#https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide 

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide/COVID-19-geographic-disbtribution-worldwide.xlsx")
#df=pd.read_csv("https://opendata.ecdc.europa.eu/covid19/casedistribution/csv")
 
df_cases = pd.DataFrame()
total_cases=np.array([])
#list the countries of interest for Covid 19 time series comparisons
country_list=['Canada','Germany','Italy','Japan','Australia']
plt.figure(figsize=(12,12)) 
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))

#figure,(ax1,ax2,ax3,ax4)=plt.subplots(2,2,figsize=(10,10))
plt.tight_layout(pad=6.0)
for i in country_list:
    df_country=df[df['countriesAndTerritories']==i]
    df_country.set_index('dateRep',inplace=True)
#plot time series of daily Covid 19 cases for countries of interest
    df_country.plot(y='cases', use_index=True,ax=ax1,title='Daily', fontsize=15)    
    ax1.legend(country_list)
    ax1.set_xlabel("Date", fontsize=15)
    ax1.set_ylabel("Covid 19 Cases", fontsize=15)
    #weekly cases of Covid 19
    weekly=df_country['cases'].resample('W').sum()
#plot time series of weekly cases of Covid 19 for countries of interest
    weekly.plot(ax=ax2,title='Weekly', fontsize=15)
    ax2.legend(country_list)
    ax2.set_xlabel("Date", fontsize=15)
    ax2.set_ylabel("Covid 19 Cases", fontsize=15)
    df_cases=df_cases.append(df_country['cases'],ignore_index = True)
    #total cases of Covid 19 for the period of study
    total=df_country['cases'].sum()
    total_cases=np.append(total_cases,total)
#clustering similar time series of daily covid 19 cases for countries of interest using linkage
#linkage is a hierarchy technique using ward method to compute distance between clusters  
z=linkage(df_cases.iloc[:,0:223],'ward')
#plot clusters of similar times series of covid 19 daily cases (note clusters with smaller distances on the y axis are more similar)
dendrogram(z,labels=country_list,leaf_rotation=45,leaf_font_size=15,ax=ax3)    
ax3.set_xlabel("Countries", fontsize=15)
ax3.tick_params(axis='both', which='major', labelsize=15)
ax3.title.set_text('Similarity of Covid 19 Daily Cases')   
#plot total number of covid 19 cases for countries of interest during the period of study
ax4.bar(range(len(total_cases)),total_cases)
ax4.set_xticks(np.arange(len(country_list)))
ax4.set_xticklabels(country_list,rotation=45, fontsize=15)
ax4.set_xlabel("Countries", fontsize=15)
ax4.set_ylabel("Total Cases of Covid 19", fontsize=15)
ax4.tick_params(axis='both', which='major', labelsize=15)









