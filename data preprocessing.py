# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:53:37 2022

@author: dhirp
"""

import pandas as pd
df=pd.read_csv("diabetes.csv")

#checking if any null value exists
df.isnull().values.any() 

#correlation check
corr=df.corr()

"""
if any two are features are corelated the remove any one feature 
using
 (del df['column name'])

if target variable is in text form then convert to number using
dict={true::1,false:0}
    df["outcome"]=df["outcome"].map(dict)
"""

colums=df.iloc[:,:2]
d=colums.values
columnnames=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
target=df["Outcome"].values
x=df[columnnames].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.impute import SimpleImputer
impute_0=SimpleImputer(missing_values=0,strategy="mean")
x_train=impute_0.fit_transform(x_train)
x_test=impute_0.fit_transform(x_test)



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)

print(knn.score(x_test, y_test))

scores={}
for i in range(1,30):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    a=knn.score(x_test, y_test)

