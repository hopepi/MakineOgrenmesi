import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/yazılımda  kullanılan boş dosyalar/kanser.csv")
veri=data.copy()


M=veri[veri["diagnosis"]=="M"]
B=veri[veri["diagnosis"]=="B"]


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kötü huylu")
plt.scatter(B.radius_mean,B.texture_mean,color="blue",label="İyi huylu")
plt.legend()
plt.show()