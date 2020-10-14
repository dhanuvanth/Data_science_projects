import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

dataSet = pd.read_csv(r'lungs\data_used_Lungs.csv')
# print the data
dataSet.head()
# summarize the data
dataSet.describe()
dataSet.drop('description', axis = 1,inplace= True)
print(dataSet)

# missing count
dataSet.iloc[:,:-1] = dataSet.iloc[:,:-1].fillna(0)
dataSet.iloc[:,:-1] = dataSet.iloc[:,:-1].replace(0,'NaN')
missing_val = (dataSet.iloc[:,:-1] == 'NaN').sum()
print(missing_val)

# Encoding dependent var
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataSet.iloc[:,1] = le.fit_transform(dataSet.iloc[:,1])
print(dataSet.iloc[:,:-1])

# independent var & dependent
X = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,-1].values
print(X)
print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy ='median')
imputer.fit(X)
X = imputer.transform(X)
print(X)

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

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=12, activation='relu'))

# Adding the third hidden layer
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

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Evaluating the Model Performance
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)