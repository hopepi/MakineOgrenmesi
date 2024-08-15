##BASİT DOĞRUSAL REGRESYON
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm##İstatistiksel modelleri tahmin etmek ve istatistiksel testler yapmamızı sağlayan bir kütüphanedir
from sklearn.linear_model import LinearRegression

data= pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/maas.csv")#excel dosyasını alma şekli
veri=data.copy()

Y=veri["Salary"]#bağımlı değişken
X=veri["YearsExperience"]#bağımsız değişken

#plt.scatter(X,Y)#noktalı grafik
#plt.show()

#stats ile oluşturma
sabit=sm.add_constant(X)#sabit parametre olmak zorunda X değerlerini yolluyoruz
model=sm.OLS(Y,sabit).fit()#fit()fonksiyonu matriks halinde sıkıştırır değerleri
print(model.summary())

#sklearn ile oluşturma
lr=LinearRegression()
lr.fit(X.values.reshape(-1,1),Y.values.reshape(-1,1))#values değerleri array yapar reshape sütun bazlı ayarlamaları vb yapar
print(lr.coef_,lr.intercept_)#parametreleri tek tek çağırabiliriz stats gibi toplu yazılamıyor fakat bir diğer sıkıntı x değerlerini array olarak yollamamız gerekiyor
#print(veri) veriler
