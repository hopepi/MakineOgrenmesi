"""
HEM RİGDE REGRESYON YAPISINI VE LASSO REGRESYON YAPISINI BİR ARADA KULLANAN MODELE ELASTİCNET REGRESYON DENİR
ELİMDE DATA YOK GENE YAZICAM
ELASTİCNET YAPISINI KÜTÜPHANESİNİ EKLEDİKTEN SONRA
elas_model=ElasticNet(alpha=0.1)
elas_model=fit(X_train,y_train)
ELASTİCNETCV YAPISINI KÜTÜPHANESİNİ EKLEDİKTEN SONRA ANLAMINI BİLİYORSUN ZATEN
lamb=ElasticNetCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_
elas_model2=ElasticNet(alpha=lamb)
elas_model2=fit(X_train,y_train)


BU ÖĞRENDİMİZ 3 ŞEYİ HEPSİNİ DENEMEMİZ GEREKİYOR EN OPTİMAL OLANI SEÇMEMİZ LAZIM TEŞEKKÜRLER
"""