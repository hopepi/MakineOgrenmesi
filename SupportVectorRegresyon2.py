"""HİSSE TAHMİN"""

import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV

data=yf.download("THYAO.IS",start="2023-09-01",end="2023-10-01")
veri=data.copy()

veri=veri.reset_index()
veri["Day"]=veri["Date"].astype(str).str.split("-").str[2]


y=veri["Adj Close"]
X=veri["Day"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)


scy=StandardScaler()
scx=StandardScaler()

X=scx.fit_transform(X)
y=scy.fit_transform(y)

svrmodel=SVR(kernel="rbf",C=1000,gamma=0.1)
svrmodel.fit(X,y)
tahminrbf=svrmodel.predict(X)


svrlinmodel=SVR(kernel="linear")
svrlinmodel.fit(X,y)
tahminlin=svrlinmodel.predict(X)


svrpolymodel=SVR(kernel="poly",degree=5)
svrpolymodel.fit(X,y)
tahminpoly=svrpolymodel.predict(X)


r2=mt.r2_score(y,tahminrbf)
rmse=mt.mean_squared_error(y,tahminrbf,squared=False)


parametreler={"C":[1,10,1000,10000],"gamma":[1,0.1,0.001,],"kernel":["rbf","linear","poly"]}##parametreleri yazıyoruz


tuning=GridSearchCV(estimator=SVR(),param_grid=parametreler,cv=10)#burda parametrelerimi tek tek denicek
tuning.fit(X,y)
print(tuning.best_params_)# en iyi parametreleri yazdıracak


print("R2: {} \nRMSE:  {}".format(r2,rmse))
plt.scatter(X,y,color="red")
plt.plot(X, tahminrbf,color="green",label="RBF MODEL")
plt.plot(X, tahminlin,color="purple",label="Linear MODEL")
plt.plot(X, tahminpoly,color="yellow",label="Poly MODEL")
plt.legend()
plt.show()
"""
EN İYİ ÇALIŞAN RBF
"""