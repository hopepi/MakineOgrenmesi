from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/yazılımda  kullanılan boş dosyalar/diyabet.csv")
veri=data.copy()


y=veri["Outcome"]
X=veri.drop(columns="Outcome",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=XGBClassifier(learning_rate=0.2,max_depth=7,n_estimators=500,subsample=0.7)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)


acs=accuracy_score(y_test,tahmin)
print(acs*100)

"""{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.7}"""
"""
parametreler={"max_depth":[3,5,7],
"subsample":[0.2,0.5,0.7],
"n_estimators":[500,1000,2000],
"learning_rate":[0.2,0.5,0.7]
}

{'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.7}"""
"""grid=GridSearchCV(model,param_grid=parametreler,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)"""