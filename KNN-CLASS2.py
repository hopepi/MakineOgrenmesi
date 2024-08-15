"""DİYELİM X1 NOKTASINDA VERİMİZ O VERİYİ KANSER OLUP OLMADIĞI ANLAMAK İÇİN ÖKLİD UZAKLIĞI EN YAKIN 5(n_neighbors DEFAULT OLARAK 5 DEĞİŞTİREBİLİRİZ) TANE KÜMESİYE ALIR VE DİYELİM 2 Sİ KANSER 1 DEĞİL BU VERİYİ KANSER DER AMA TAM TERSİ İSE DEĞİL DER """
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/yazılımda  kullanılan boş dosyalar/kanser.csv")
veri=data.copy()


veri=veri.drop(columns=["id","Unnamed: 32"],axis=1)


veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]


y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=KNeighborsClassifier(n_neighbors=9)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)


acs=accuracy_score(y_test,tahmin)
print(acs*100)


#DEFAULT OLAN n_neighbors=5 PARAMETRESİNİN OPTİMAL SONUCUNU ARAMA ALGORİTMASI

"""
basarı=[]


for k in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    tahmin2=knn.predict(X_test)
    basarı.append(accuracy_score(y_test,tahmin2))


plt.plot(range(1,20),basarı)
plt.xlabel("K")
plt.ylabel("Başarı")
plt.show()
"""