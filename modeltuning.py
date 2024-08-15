import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt


data= pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/reklam.csv")#excel dosyasını alma şekli
veri=data.copy()
veri.drop("Unnamed: 0",axis=1,inplace=True)

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=150)##random state(hiper parametre örneği) veriyi nerden kesiceni belirler modelden modele göre değişir

lr=LinearRegression()
model=lr.fit(X_train,y_train)


def skor(model,x_train,x_test,y_train,y_test):
    egitimtahmin=model.predict(x_train)
    testtahmin=model.predict(x_test)


    r2_egitim=mt.r2_score(y_train,egitimtahmin)
    r2_test=mt.r2_score(y_test,testtahmin)

    mse_egitim=mt.mean_squared_error(y_train,egitimtahmin)
    mse_test=mt.mean_squared_error(y_test,testtahmin)


    return[r2_egitim,r2_test,mse_egitim,mse_test]

sonuc1= skor(model=lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)

print("Eğitim R2= {} Eğitim MSE={}".format(sonuc1[0],sonuc1[2]))
print("Test R2= {} Test MSE={}".format(sonuc1[1],sonuc1[3]))

##Eğitim datasetini doğrulama yaparak bölmek
lr_cv=LinearRegression()
k=5
iteratasyon=1
cv=KFold(n_splits=k)

for egitimindex,tesindex in cv.split(X):
    X_train,X_test=X.loc[egitimindex],X.loc[tesindex]
    y_train, y_test = y.loc[egitimindex], y.loc[tesindex]
    lr_cv.fit(X_train,y_train)

    sonuc2=skor(model=lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
    ###AMACIMIZ DAHA YÜKSEK R2 DEĞERİ DAHA DÜŞÜK MSE DEĞERİNİ BULMAK
    print("İteratasyon {}".format(iteratasyon))###AMACIMIZ DAHA YÜKSEK R2 DEĞERİ DAHA DÜŞÜK MSE DEĞERİNİ BULMAK
    print("Eğitim R2= {} Eğitim MSE={}".format(sonuc2[0], sonuc2[2]))###AMACIMIZ DAHA YÜKSEK R2 DEĞERİ DAHA DÜŞÜK MSE DEĞERİNİ BULMAK
    print("Test R2= {} Test MSE={}".format(sonuc2[1], sonuc2[3]))###AMACIMIZ DAHA YÜKSEK R2 DEĞERİ DAHA DÜŞÜK MSE DEĞERİNİ BULMAK
    iteratasyon+=1###AMACIMIZ DAHA YÜKSEK R2 DEĞERİ DAHA DÜŞÜK MSE DEĞERİNİ BULMAK
###AMACIMIZ DAHA YÜKSEK R2 DEĞERİ DAHA DÜŞÜK MSE DEĞERİNİ BULMAK


"""
polinomal regresyonda vardır videosunu izlersin
"""