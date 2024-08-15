"""
Rigde regresyondaki alpha değeri 0 olduğu durumu ele alır Lasso yapısı modelden dışlama yapılıyor
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
"""
Load_boston yapısı kaldırılmış ondan not alıcam farklı bişi öğreticem
Eğer eğitim yapısı %99 gibi güzel bir değer çıkıp ama aynı şekilde test verisinin bir o kadar kötü çıkması aşırı öğrenme şüphesi vardır
linear modelden Lassoyu eklememiz gerek 
lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
Burda herhangi bir parametreyi dışlar değerlerin düşmesi beklenir
LassoCV kütüphanesi ekleriz
lamb=LassoCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_
#en doğru değeri bulur
lasso_model=Lasso(alpha=lamb)
lasso_model.fit(X_train,y_train)
"""

