# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:56:46 2020

@author: adehg
"""
# The code is developed by adehghanGitHub in September 2020 to predict housing price.
#data source at UCI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import model_selection

df=pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real estate valuation data set.xlsx')
#data analysis of the model target and features
df['Y house price of unit area'].describe()
df['Y house price of unit area'].plot.hist(bins=50)
list_of_features=list(df.drop(['Y house price of unit area'],axis=1).columns)
#correlation of between target and features
corr=(df.drop(['No','X1 transaction date'],axis=1).corr()['Y house price of unit area'])
plt.figure(figsize=(20,10))
#removing NO and transaction date
df_no_date=df.drop(['No','X1 transaction date'],axis=1)
#plot correlation of data features
sns.heatmap(df_no_date.corr(),xticklabels=df_no_date.columns,yticklabels=df_no_date.columns, annot=True)

x=df.drop(['No','X1 transaction date','Y house price of unit area'],axis=1)#features
y=df['Y house price of unit area']#target
#split data to training and test data sets
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
count_nan_x = x_train.isna().sum()
count_nan_y=y_train.isna().sum()
#model feature selection
regressor = RandomForestRegressor(max_depth=12, n_estimators = 300,random_state= 0)
regressor.fit(x_train,y_train)
cols=x.columns
feature_imp = pd.Series(regressor.feature_importances_,index=cols).sort_values(ascending=False)
plt.figure(figsize=(8,10))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features (RF)");
xx=df[['X3 distance to the nearest MRT station','X2 house age']]
yy=df['Y house price of unit area']

#model selection
model=[LinearRegression(),KNeighborsRegressor(),RandomForestRegressor()]
results=[]

for m in model:
    kfold=model_selection.KFold(n_splits=10,random_state=7)
    model=m.fit(x_train, y_train)
    crossVal=model_selection.cross_val_score(model,xx,yy,scoring='neg_mean_squared_error',cv=kfold)
    results.append(crossVal.mean())
best_regressor=model[np.argmax(results)]
#apply the best regressor
x_train, x_test, y_train, y_test=train_test_split(xx,yy,test_size=0.2)
regressor = RandomForestRegressor(max_depth=12, n_estimators = 300,random_state= 0)
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)    
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)



    
    
