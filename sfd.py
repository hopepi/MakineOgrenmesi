"""
hataları vb düzeltme
eksik gözlem olursa ne olur ?
veri.isnull().sum() yaptığımızda 0 dan fazla ise değerler düzenleme gerektirebilir
null değerlerini doldurmak için yardımcı kütüphaneden faydalanırım bu da from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy"mean") burda numpydan null değerleri alarak ortalama değere eşitledik
imputer=imputer.fit(veri) veriyi fit ettik
daha sonra değeri ıloc kullanrak veri.iloc[:,:]=imputer.transform(veri)



modeli oluşturduktan sonra hata ayıklamak için import sklearn.metrics as mt

r2=mt.r2_score(y_test,tahmin)
mse=mt.mean_squared_error(y_test,tahmin)
rmse=mt.mean_squared_error(y_test,tahmin,squared=False)
mae=mt.mean_absolute_error(y_test,tahmin)
"""
"""
#model tuning
hiperparametre radyo örneğinde olduğu gibi daha net ses duymak için radyo düğmesinde frekansıyla oynayınca o düğme hiperparametre olur


"""