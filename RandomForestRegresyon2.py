import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
import sklearn.metrics as mt

data=pd.read_csv("C:/Users/umutk/OneDrive/Masaüstü/reklam.csv")
veri=data.copy()


y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)


dtmodel=DecisionTreeRegressor(random_state=0,max_leaf_nodes=19,min_samples_split=2)
dtmodel.fit(X_train,y_train)
dttahmin=dtmodel.predict(X_test)


bgmodel=BaggingRegressor(random_state=0,n_estimators=19)
bgmodel.fit(X_train,y_train)
bgtahmin=bgmodel.predict(X_test)


rftmodel=RandomForestRegressor(random_state=0,max_leaf_nodes=19,min_samples_split=9,n_estimators=9)
rftmodel.fit(X_train,y_train)
rftahmin=rftmodel.predict(X_test)


r2dt=mt.r2_score(y_test,dttahmin)
r2bg=mt.r2_score(y_test,bgtahmin)
r2rft=mt.r2_score(y_test,rftahmin)


rmsedt=mt.mean_squared_error(y_test,dttahmin,squared=False)
rmsebg=mt.mean_squared_error(y_test,bgtahmin,squared=False)
rmserft=mt.mean_squared_error(y_test,rftahmin,squared=False)


print("KARAR AĞACI MODELİ = R2 : {}  RMSE : {}".format(r2dt,rmsedt))
print("BAG MODEL = R2 : {}  RMSE : {}".format(r2bg,rmsebg))
print("RANDOM FOREST MODELİ = R2 : {}  RMSE : {}".format(r2rft,rmserft))


"""dtparametreler={"min_samples_split":range(2,20),"max_leaf_nodes":range(2,20)}
dtgrid=GridSearchCV(estimator=dtmodel,param_grid=dtparametreler,cv=10)#n_jobs=-1 yazılıcak yer
dtgrid.fit(X_train,y_train)
print(dtgrid.best_params_)


bgparametreler={"n_estimators":range(2,20)}
bggrid=GridSearchCV(estimator=bgmodel,param_grid=bgparametreler,cv=10)#n_jobs=-1 yazılıcak yer
bggrid.fit(X_train,y_train)
print(bggrid.best_params_)


rftparametreler={"min_samples_split":range(2,20),"max_leaf_nodes":range(2,20),"n_estimators":range(2,20)}
rftgrid=GridSearchCV(estimator=rftmodel,param_grid=rftparametreler,cv=10)#n_jobs=-1 yazılıcak yer
rftgrid.fit(X_train,y_train)
print(rftgrid.best_params_)"""


"""ÇIKTISI 10 DAKİKA SÜRDÜ XD
{'max_leaf_nodes': 19, 'min_samples_split': 2}
{'n_estimators': 19}
{'max_leaf_nodes': 19, 'min_samples_split': 9, 'n_estimators': 9}
"""


"""                                                     ÖNEMLİ BİR NOT
BURDA RANDOM FOREST YAPISI KOMBİNASYON DENEDİĞİNDEN UYUMLU PARAMETRE BULMASI 5 10 DAKİKA BELKİDE DAHA FAZLA OLABİLİR SABIRLI OL
AMA İŞİ HIZLANDIRMAK İSTİYORSAN İŞLEMCİNİ TAM YÜK ALTINDA ÇALIŞTIRICAK BİR FONKSİYON KULLANABİLİRİSİN O FONKSİYONDA == n_jobs=-1 dur 
bunu gridsearchcv yapısına yazıcaksın
"""