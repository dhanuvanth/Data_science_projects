import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm

df = pd.read_excel(r'Truck Analysis.xlsx', sheet_name='Data')

print(df.describe())

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['From'] = le.fit_transform(df['From State'])
df['To'] = le.fit_transform(df['To state'])
df['Type'] = le.fit_transform(df['vehicleType'])
df['delay'] = le.fit_transform(df['delay'])
print(df)

dataSet = df[['From','To','Type','No of days','TRANSPORTATION_DISTANCE_IN_KM','delay']]
print(dataSet)

X = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,-1].values
print(X)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Train
# from xgboost import XGBClassifier
# classifier = XGBClassifier()
# classifier.fit(X_train,y_train)

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, y_train)

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)
# classifier.fit(X_train, y_train)

# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)

# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
model.fit(X_train, y_train)

# from sklearn.model_selection import GridSearchCV

# model = SVC()

# parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5,n_jobs=-1)
# grid.fit(X_train, y_train)

# roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
# print('Score: {}'.format(roc_auc))

# model = RandomForestClassifier(n_estimators=1000)
# model.fit(X_train, y_train)
# predictions = cross_val_predict(model, X_test, y_test, cv=5)
# print(classification_report(y_test, predictions))

# print(confusion_matrix(y_test, predictions))

# score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
# np.around(score, decimals=4)

print(model.predict(sc.transform([[7,15,35,3,200]])))

# Pridict test Results
y_pred = model.predict(X_test)
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
accuracies = cross_val_score(estimator= model,X = X_train,y = y_train,cv = 2,scoring='accuracy')
print('Accuracy : {:.2f} %'.format(accuracies.mean() * 100))
print('Standard Deviation : {:.2f} %'.format(accuracies.std() * 100))


# backward elumination
x_opt = X
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()  
print(regressor_OLS.summary())