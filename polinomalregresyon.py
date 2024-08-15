import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as mt


data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/Ev.csv")
veri=data.copy()

veri.drop(columns=["No","X1 transaction date","X5 latitude","X6 longitude"],axis=1,inplace=True)

"""SÜTUNLARIN İSİMLERİNİ DEĞİŞTİRME"""
veri=veri.rename(columns={"X2 house age":"Ev yasi",
"X3 distance to the nearest MRT station":"Metroya uzaklik",
"X4 number of convenience stores":"Market sayisi",
"Y house price of unit area":"Ev fiyati"
})


y=veri["Ev fiyati"]
X=veri.drop(columns="Ev fiyati",axis=1)


pol=PolynomialFeatures(degree=3)#degree polinomun derecesi oluyor hiperparametredir
X_pol=pol.fit_transform(X)#2 dereceden polinomu oluşturduk

X_train,X_test,y_train,y_test=train_test_split(X_pol,y,test_size=0.2,random_state=42)

pol_reg=LinearRegression()
pol_reg.fit(X_train,y_train)
tahmin=pol_reg.predict(X_test)

r2=mt.r2_score(y_test,tahmin)
Mse=mt.mean_squared_error(y_test,tahmin)

print("R2:  {} MSE:  {}".format(r2,Mse))#değeri optimal değere yaklaştırmak için polinomun derecesiyle oynamalıyız (2 den 3 e yükselttim daha optimal oldu)

