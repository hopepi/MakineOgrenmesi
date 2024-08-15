import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/pozisyon.csv")
veri=data.copy()

y=veri["Salary"]
X=veri["Level"]

y=np.array(y).reshape(-1,1)
X=np.array(X).reshape(-1,1)


dtr=DecisionTreeRegressor(random_state=0,max_leaf_nodes=5)#max_leaf_nodes koymadan önce aşırı uyum yani makine ezberliyordu max_leaf_nodes aslında ağacın kaç dalı olucağına karar veriyor
dtr.fit(X,y)
tahmin=dtr.predict(X)


"""plt.scatter(X,y,color="red")
plt.plot(X,tahmin)
plt.show()"""


plt.figure(figsize=(20,10),dpi=100)
plot_tree(dtr,feature_names="Level",class_names="Salary",rounded=True,filled=True)
plt.show()