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
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

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