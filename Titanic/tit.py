import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv(r'Titanic\train.csv')
test = pd.read_csv(r'Titanic\test.csv')
# print the data
train.head()
test.head()
# summarize the data
train.describe()
test.describe()
train.drop(['Name','Ticket','Cabin','Embarked'], axis = 1,inplace= True)
test.drop(['Name','Ticket','Cabin','Embarked'], axis = 1,inplace= True)
print(train)
print(test)

missing_val_train = train.isnull().sum()
missing_val_test = test.isnull().sum()
print(missing_val_train)
print(missing_val_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train.iloc[:,3] = le.fit_transform(train.iloc[:,3])
test.iloc[:,2] = le.fit_transform(test.iloc[:,2])
print(train)
print(test)

# Take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy ='median')
imputer.fit(train.iloc[:,2:])
imputer.fit(test.iloc[:,1:])
train.iloc[:,2:] = imputer.transform(train.iloc[:,2:])
test.iloc[:,1:] = imputer.transform(test.iloc[:,1:])
print(train)


# Encoding Indipendent var
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1,4,5])],remainder='passthrough')
train = ct.fit_transform(train)
test = ct.fit_transform(test)

X_train_1 = train.iloc[:,2:].values
X_test_1 = test.iloc[:,1:].values
y_train_1 = train.iloc[:, 1].values
print(X_train_1)
print(X_test_1)
print(y_train_1)


# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_train_1,y_train_1,test_size = 0.2,random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_1 = sc.fit_transform(X_test_1)
print(X_train)
print(X_test)
print(X_test_1)

# # Train
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

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)