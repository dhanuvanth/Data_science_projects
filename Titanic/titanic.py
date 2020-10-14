import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# get Dataset
train = pd.read_csv(r"Titanic\train.csv")
test = pd.read_csv(r"Titanic\test.csv")
X_train = train.iloc[:,[0,2,4,5,6,7,9]].values
X_test = test.iloc[:,[0,1,3,4,5,6,8]].values
y_train = train.iloc[:, 1].values
print(X_train)
print(y_train)

# Take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy ='median')
imputer.fit(X_train[:,3:])
imputer.fit(X_test[:,3:])
X_train[:,3:] = imputer.transform(X_train[:,3:])
X_test[:,3:] = imputer.transform(X_test[:,3:])
print(X_train)

# Encoding Indipendent var
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)
print(X_train)
print(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,1:] = sc.fit_transform(X_train[:,1:])
X_test[:,1:] = sc.transform(X_test[:,1:])

# Train
# from xgboost import XGBClassifier
# classifier = XGBClassifier()
# classifier.fit(X_train,y_train)
# # from sklearn.naive_bayes import GaussianNB
# # classifier = GaussianNB()
# # classifier.fit(X_train, y_train)
# # from sklearn.ensemble import RandomForestClassifier
# # classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
# # classifier.fit(X_train, y_train)
# # from sklearn.neighbors import KNeighborsClassifier
# # classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# # classifier.fit(X_train, y_train)
# # from sklearn.svm import SVC
# # classifier = SVC(kernel = 'linear', random_state = 0)
# # classifier.fit(X_train, y_train)

# # Pridict test Results
# y_pred = classifier.predict(X_train)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_train.reshape(len(X_train),1)),1))

#Initialization
ann = tf.keras.models.Sequential()

#add neutron 1
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))

#add neutron 2
ann.add(tf.keras.layers.Dense(units = 12,activation = 'relu'))

#add neutron 3
ann.add(tf.keras.layers.Dense(units = 12,activation = 'relu'))

#add neutron 4
ann.add(tf.keras.layers.Dense(units = 12,activation = 'relu'))

#add neutron 5
ann.add(tf.keras.layers.Dense(units = 6,activation = 'relu'))

#output neutron
ann.add(tf.keras.layers.Dense(units = 1,activation = 'sigmoid'))

#compile
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training
ann.fit(X_train,y_train,batch_size = 32,epochs = 100)

#prediction
# print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])))

# Pridict test Results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), X_test.reshape(len(X_test),1)),1))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
print(cm)

# Evaluating the Model Performance
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, y_pred)
print(accuracy)


print(ann.predict(X_train))

submission=pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = ann.predict(X_test)
submission.to_csv('submissionrd.csv',index=False)