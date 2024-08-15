import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
import sklearn.metrics as mt
import numpy as np

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/reklam.csv")
veri=data.copy()

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


lr=LinearRegression()
lr.fit(X_train,y_train),
tahmin=lr.predict(X_test)


r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
print("R2:  {} MSE:  {}".format(r2,mse))
"""
ÇOK DEĞİŞKEN(SÜTUN) OLDUĞUNDA BAZEN;
R2 DEĞERİ YÜKSEK OLABİLİR FAKAT BAZEN BU SAHTE GÖRÜNÜM OLABİLİR RİDGE YAPISI KATSAYILARINI CEZALANDIRIR VE R2 DEĞERİNİ OPTİMAL GERÇEK SEVİYESİNE GETİRİR 
"""
rigde_model=Ridge(alpha=0.1)#ridge yapısını burda ayarlıyoruz alphayı
rigde_model.fit(X_train,y_train)
tahmin2=rigde_model.predict(X_test)

r2rid=mt.r2_score(y_test,tahmin2)
mserid=mt.mean_squared_error(y_test,tahmin2)
print("R2:  {} MSE:  {}".format(r2rid,mserid))
#alphayı bilmediğimizden bu beklediğimizden bundan sonrasında mantık biraz zor video 2 yi tekrar izleyebilirsin
"""
VERİ SETİNE ERİŞİM YOK NOT ALARAK YAPICAM BUNU
numpydan alpha değerleri oluşuruyoruz çünkü hangisi daha iyi bilmiyoruz
lambdalar=10**np.linspace(10,-2,100)*0.5
rigdecv=RigdeCv(alphas=lambdalar,scoring="r2")#bu kütüphanedeki fonksiyon bizim oluşturduğumuz lambdalar serisinin içinden en optimal olanı buluyor
rigde_cv.fit(X_train,y_train)
alpha değerini bir değişkenle alıp yukarıdaki alpha değerine yapıştırabiliriz
print(rigde_cv.alpha_)#değeri kopyalayıp alphaya yapıştır
"""
