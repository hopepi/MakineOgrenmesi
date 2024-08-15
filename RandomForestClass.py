import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/yazılımda  kullanılan boş dosyalar/diyabet.csv")
veri=data.copy()


y=veri["Outcome"]
X=veri.drop(columns="Outcome",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=RandomForestClassifier(random_state=0)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)


acs=accuracy_score(y_test,tahmin)
print(acs*100)