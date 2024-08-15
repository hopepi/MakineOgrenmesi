import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/sarap.csv")
veri=data.copy()


y=veri["quality"]
X=veri.drop(columns="quality",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

"""
çalıştığımız değişken sayısını azaltmamızı sağlar fakat bunu yaparken önemsiz veri kaybı bekleriz eğer transform ettikten sonra kontrol yapmak lazım eğer azalma yoksa şimdi pca içini doldurucağım gibi yapıcaz
"""
pca=PCA()#şimdilik değer vermicem n_components=2 denedim daha önce

X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

print(np.cumsum(pca.explained_variance_ratio_)*100)
"""
Burda çıkan 11 değerde değişken sayına göre bu değişir genel olarak %90 üzerindekileri muhattap olcak çünkü %90 altındakiler çok düşük bir değer
ve fedakarlıktan sonra sallıyorum 9 değişkenle çalışmaya devam etmeye çalışcam dedim PCA() içine PCA(n_components=9) yazıcam misal
"""

lm=LinearRegression()
lm.fit(X_train2,y_train)
tahmin=lm.predict(X_test2)


r2=mt.r2_score(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=True)

print("R2 :",r2.__str__() +"  RMSE :",rmse.__str__())

##çapraz doğrulama testi
cv=KFold(n_splits=10,shuffle=True,random_state=1)##n_splits genelde 10 oluyor shuffle değişkenleri karıştırma oluyor

lm2=LinearRegression()
RMSE=[]

for i in range(1,X_train2.shape[1]+1):
    hata=np.sqrt(-1*cross_val_score(lm2,X_train2[:,:i],y_train.ravel(),cv=cv,scoring="neg_mean_squared_error").mean())
    RMSE.append(hata)



## kırılgan noktalarına dikkat et
plt.plot(RMSE,"-x")
plt.xlabel("Bileşen sayısı")
plt.ylabel("RMSE")
plt.show()
"""LDA"""
"""
LDA YAPISI SINIFLAR ARASINDAKİ BOŞLUĞU AÇMAK İÇİN KULLANILIR

lda=LinearDiscriminantAnalysis()# herhangi değer girmezsem kendisi 5i otomatik yapıyor (n_components=5) yani
X_train3=lda.fit_transform(X_train,y_train)
X_test3=lda.fit_transform(X_test)
print(len(np.unique(y_train)))#bu değerin -1 kadarıyla n_components maks değerini bulmuş olursun
print(np.cumsum(pca.explained_variance_ratio_)*100)##11 değer çıkmıyor çünkü lda yapısı sınıflandırıyor içinde
"""