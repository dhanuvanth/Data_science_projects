import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# get Dataset
dataSet = pd.read_csv(r"XGBoost\diabetes.csv")
X = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:, -1].values

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Train
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

# Pridict test Results
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Evaluating the Model Performance
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# k-Fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= classifier,X = X_train,y = y_train,cv = 10,scoring='accuracy')
print('Accuracy : {:.2f} %'.format(accuracies.mean() * 100))
print('Standard Deviation : {:.2f} %'.format(accuracies.std() * 100))