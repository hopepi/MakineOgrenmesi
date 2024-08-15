"""
RANDOM FOREST AŞIRI ÖĞRENME DURUMUNA DÜŞMEMEK İÇİN GENELDE KULLANILAN YÖNTEM AMA SEKTÖRDE BU DAHA ÇOK KULLANILIYOR VE BAGGİNGDEN FARKI HER Bİ MODEL İÇİNDEKİ HATALARI AĞIRLAKLANDIRILMADIĞI İÇİN PROBLEM BURDA BAŞLIYOR
BAGGİNG GİBİ DEĞİŞKENLERİ RASTGELE ALIYOR DAHA SONRA DURMAYIP RASTGELE SÜTUNLARDA ALIP BOOTSTRAP YAPISI YAPIYOR MODELİ ÇIKARIYOR
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/pozisyon.csv")
veri=data.copy()


y=veri["Salary"]
X=veri["Level"]


y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)


dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X,y)
dttahmin=dtmodel.predict(X)


rftmodel=RandomForestRegressor(random_state=0)
rftmodel.fit(X,y)
rftahmin=rftmodel.predict(X)


plt.scatter(X,y,color="red")
plt.plot(X,dttahmin,color="blue")#AŞIRI ÖĞRENME GERÇEKLEŞTİ
plt.plot(X,rftahmin,color="green")
plt.show()