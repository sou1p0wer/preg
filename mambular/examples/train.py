import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
import sys
sys.path.append('/data/high_risk_pregnant/mambular')

from mambular.models import MambularClassifier

data = pd.read_csv('/data/high_risk_pregnant/pregnant/data/bishe_data_test/val.csv',low_memory=False)
data = data.iloc[:1000,:]
# convert_dict = {
#     'premature':float,
#     'low_BW':float,
#     'macrosomia':float,
#     'death':float,
#     'malformation':float,
#     '分娩方式':float,
#     '产后出血':float,
#     'SGA_S1':int, 
#     'LGA_S1':int
# }
# data = data.astype(convert_dict)
X = data.drop(columns=['premature', 'low_BW', 'macrosomia', 'death', 'malformation','分娩方式','产后出血','SGA_S1', 'LGA_S1'])

y1 = data["premature"].values
y1 = y1.reshape((-1,1))
y2 = data["low_BW"].values
y2 = y2.reshape((-1,1))
y3 = data["macrosomia"].values
y3 = y3.reshape((-1,1))
y4 = data["death"].values
y4 = y4.reshape((-1,1))
y5 = data["malformation"].values
y5 = y5.reshape((-1,1))
y6 = data["分娩方式"].values
y6 = y6.reshape((-1,1))
y7 = data["产后出血"].values
y7 = y7.reshape((-1,1))
y8 = data["SGA_S1"].values
y8 = y8.reshape((-1,1))
y9 = data["LGA_S1"].values
y9 = y9.reshape((-1,1))
y = np.concatenate([y1,y2,y3,y4,y5,y6,y7,y8,y9],axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Instantiate the classifier
classifier = MambularClassifier()
# Fit the model on training data
classifier.fit(X=X_train,y=y_train, max_epochs=1, batch_size=16)
# print(classifier.evaluate(X_test, y_test))
