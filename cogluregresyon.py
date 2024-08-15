##ÇOKLU REGRESYON
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split#eğitim verisi ayırma
from sklearn.linear_model import LinearRegression



data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/reklam.csv")
veri=data.copy()

veri=veri.drop(["Unnamed: 0"],axis=1)

print(veri.corr()["Sales"])
#print(veri.dtypes)

Q1=veri["Newspaper"].quantile(0.25)#gazetede uyumsuzluk olduğu için baskılama işlemi
Q3=veri["Newspaper"].quantile(0.75)#gazetede uyumsuzluk olduğu için baskılama işlemi
IQR=Q3-Q1
ustsınır=Q3+1.5*IQR#ust sınır hesaplama
aykırı=veri["Newspaper"]>ustsınır#aykırı değerleri alma
veri.loc[aykırı,"Newspaper"]=ustsınır

y=veri["Sales"]
X=veri[["TV","Radio"]]#Newspaper coef katsayısı - çıktığı için Newspaperda bir anlamsızlık var bu değeri dışlamamız gerek olmasaydı Newspaperıda eklerdik

#sabit =sm.add_constant(X)
#model=sm.OLS(y,sabit).fit()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)#Genelde test ve eğitim verisi eğitim için %80 ayrılır fakat büyük veriler için bu artabilir testsize 0.2 yaparak eğitim verisini %80 nini ayrımış oluruz randomstate değeri ise imgesel 42 herhangi bir değerde verebiliriz

lr=LinearRegression()
lr.fit(X_train,y_train)#makine öğretiyoruz

tahmin=lr.predict(X_test)#ortaya y tahmin değeri çıkartıyor burda verileri yolladık
y_test=y_test.sort_index()#bu sıralamayı yapmaz isek çok karışık bir tablo çıkar karşımıza

#print(tahmin)#y tahmin değerleri

df=pd.DataFrame({"Gerçek":y_test,"Tahmin":tahmin})
df.plot(kind="line")# karşılaştırma
plt.show()