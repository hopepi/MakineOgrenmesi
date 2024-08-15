import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/pozisyon.csv")
veri=data.copy()


y=veri["Salary"]
X=veri["Level"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)


scx=StandardScaler()
scy=StandardScaler()

y=scy.fit_transform(y)
X=scx.fit_transform(X)

svrmodel=SVR(kernel="rbf")
"""burada kullandığımız kernel parametresinin karşılığı ‘rbf’ zaten varsayılan çekirdek. Bu şu demek: değişkenler arasındaki ilişki durumuna göre bir çekirdek seçilmeli"""
svrmodel.fit(X,y)
tahmin=svrmodel.predict(X)


plt.scatter(X,y,color="red")
plt.plot(X,tahmin)
plt.show()