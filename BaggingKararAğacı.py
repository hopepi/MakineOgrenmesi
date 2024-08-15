""" BAGGİNG AŞIRI ÖĞRENME DURUMUNA DÜŞMEMEK İÇİN GENELDE KULLANILAN YÖNTEM"""

import pandas as pd
import sklearn.metrics as mt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import BaggingRegressor

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/reklam.csv")
veri=data.copy()


y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


dtr=DecisionTreeRegressor(random_state=0,max_leaf_nodes=24,min_samples_split=6)
dtr.fit(X_train,y_train)
tahmin=dtr.predict(X_test)


r2=mt.r2_score(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=False)


print("R2 :",r2.__str__() +"  RMSE :",rmse.__str__())


bgmodel=BaggingRegressor(random_state=0,n_estimators=21)
bgmodel.fit(X_train,y_train)
tahmin2=bgmodel.predict(X_test)


r22=mt.r2_score(y_test,tahmin2)
rmse2=mt.mean_squared_error(y_test,tahmin2,squared=False)


print("BAGGİNG R2 :",r22.__str__() +" BAGGİNG  RMSE :",rmse2.__str__())


"""PARAMETRE DENEME"""
"""parametreler={"min_samples_split":range(2,25),"max_leaf_nodes":range(2,25)}
grid1=GridSearchCV(estimator=dtr,param_grid=parametreler,cv=10)
grid1.fit(X_train,y_train)
print(grid1.best_params_)


parametreler2={"n_estimators":range(2,25)}
grid2=GridSearchCV(estimator=bgmodel,param_grid=parametreler2,cv=10)
grid2.fit(X_train,y_train)
print(grid2.best_params_)"""