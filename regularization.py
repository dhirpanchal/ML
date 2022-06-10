# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:05:07 2022

@author: dhirp
"""

import pandas as pd
import numpy as np
url="https://raw.githubusercontent.com/edyoda/ML-with-Rishi/main/Detail_Cars.csv"
df=pd.read_csv(url)

df=df.replace("?",np.nan)
obj_cols=df.select_dtypes(include="object")

df['price']=pd.to_numeric(df["price"],errors="coerce")
df['peak-rpm']=pd.to_numeric(df["peak-rpm"],errors="coerce")
df['horsepower']=pd.to_numeric(df["horsepower"],errors="coerce")
df['stroke']=pd.to_numeric(df["stroke"],errors="coerce")
df['bore']=pd.to_numeric(df["bore"],errors="coerce")

obj_cols2=df.select_dtypes(include="object")     `                                                   

df=df.drop("normalized-losses",axis=1)

print(df["num-of-cylinders"].unique())
df["num-of-cylinders"].replace({"two":2,"three":3,"four":4,"five":5,"six":6,"eight":8,
                               "twelve":12},inplace=True)

df=pd.get_dummies(df,drop_first=True)
df=df.dropna()
Corr=df.corr()
 
 
x=df.drop("price",axis=1)
y=df["price"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

cols=df.columns
coef= pd.Series(model.coef_,cols[1:]).sort_values()

coef.plot(kind="bar")

from sklearn.linear_model import Lasso
lasso=Lasso(alpha=5,normalize=(True))
lasso.fit(x_train,y_train)
print(lasso.score(x_test,y_test))
print(lasso.score(x_train,y_train))

lasso_coef=pd.Series(lasso.coef_,cols[1:]).sort_values()
lasso_coef.plot(kind='bar')









