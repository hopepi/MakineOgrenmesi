import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/yazılımda  kullanılan boş dosyalar/kanser.csv")
veri=data.copy()


veri=veri.drop(columns=["id","Unnamed: 32"],axis=1)


veri.diagnosis=[1 if kod=="M" else 0 for kod in veri.diagnosis]


y=veri["diagnosis"]
X=veri.drop(columns="diagnosis",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model =DecisionTreeClassifier(random_state=0,criterion="gini",max_depth=6,max_leaf_nodes=9,min_samples_split=2)
model.fit(X_train,y_train)
tahmin=model.predict(X_test)


acs=accuracy_score(y_test,tahmin)
print(acs*100)

"""{'criterion': 'gini', 'max_depth': 6, 'max_leaf_nodes': 9, 'min_samples_leaf': 3, 'min_samples_split': 2}"""


"""parametreler={"criterion":["gini","entropy","log_loss"],
"max_leaf_nodes":range(2,10),
"max_depth":range(2,10),
"min_samples_split":range(2,10),
"min_samples_leaf":range(2,10)}


grid=GridSearchCV(model,param_grid=parametreler,cv=10)
grid.fit(X_train,y_train)
print(grid.best_params_)"""