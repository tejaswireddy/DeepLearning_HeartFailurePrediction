# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

#converting dateTime to milliseconds
#import time
#from datetime import datetime
#dt = datetime(2018, 1, 1)
#milliseconds = int(round(dt.timestamp() * 1000))
#print(milliseconds)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#script_dir = os.path.dirname(__file__)
#abs_file_path = os.path.join(script_dir, 'example.csv')

# Importing the dataset
dataset = pd.read_csv('training-data-ff.csv')
dataset = dataset.dropna()
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 4].values

#X = X[~np.isnan(X)]
#y = y[~np.isnan(y)]

#Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoder_X_1 = LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#labelencoder_X_2 = LabelEncoder()
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#there is no relational order between out categorical variables that means one diagnosis code is not higher etc
#this will create the dummy variables
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()
#X = X[:, 1:]

#to remove one dummy variable column not needed for us as of now
#X = onehotencoder.fit_transform(X).toarray()
#X= X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling - compulsory
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras import backend
from tensorflow.contrib.keras import optimizers

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=3))

# Adding the second hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#classifier.compile(loss='mean_squared_error', optimizer=sgd)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=5, epochs=10)


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
backend.clear_session()

#predict a single dataset/single observation
new_prediction = classifier.predict(sc.transform(np.array([[9867856477, 641.9, 1519689600]])))
new_prediction  = (new_prediction  > 0.5)

#Part 4 Evaluating, Improving and Tuning the ANN
#k fold cross validation
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=3))
    classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier    
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 10)
accuracies = cross_val_score(estimator = classifier,  X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

#improving the ANN





