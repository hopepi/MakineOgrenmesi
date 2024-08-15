"""Dondurma Örneği"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
import sklearn.metrics as mt

data=pd.read_csv("https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv")
veri=data.copy()


y=veri["Revenue"]
X=veri["Temperature"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=DecisionTreeRegressor(random_state=0,max_leaf_nodes=21,min_samples_leaf=17)
model.fit(X_train.values.reshape(1,-1),y_train.values.reshape(1,-1))
tahmin=model.predict(X_test)


r2=mt.r2_score(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=True)


print("R2 :",r2.__str__() +"  RMSE :",rmse.__str__())


parametreler={"min_samples_split":range(2,50),"max_leaf_nodes":range(2,50)}


grid=GridSearchCV(estimator=model,param_grid=parametreler,cv=10)
grid.fit(X_train.values.reshape(1,-1),y_train.values.reshape(1,-1))
print(grid.best_params_)
"""
HATA ALICAM YAPMAYA ÜŞENDİM ÇÖZÜMÜ İKİ BOYUTLU SERİYE ÇEVİR
"""