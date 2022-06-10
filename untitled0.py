# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:54:31 2022

@author: dhirp
"""

from sklearn.datasets import load_diabetes
data=load_diabetes()
dfx=data.data
dfy=data.target

import pandas as pd
x=pd.DataFrame(data.data,columns=data.feature_names).values
y=pd.DataFrame(data.target,columns=["Progress"]).values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

import matplotlib.pyplot as plt
plt.scatter(range(1,443),dfy)
plt.show()

scores={}
for n in range(1,30):
    from sklearn.neighbors import KNeighborsRegressor
    knn=KNeighborsRegressor(n_neighbors=n)
    knn.fit(x_train,y_train)
    
    y_pred=knn.predict(x_test)
    scores[n]=knn.score(x_test,y_test)
  #  print(knn.score(x_test,y_test))


from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=knn.predict(x_test)

print(model.score(x_test,y_test))

