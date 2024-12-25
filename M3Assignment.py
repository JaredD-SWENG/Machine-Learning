# Your name: Jared Daniel
# Your PSU Email: jjd6385

# Assignment name: Decision Tree
# Module number: 3

from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import pandas as pd

# load the Iris dataset with pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'label']
dataset = pd.read_csv(url, names=names)

# Data set
X = dataset.values[:,0:4].astype(float)
# Labels
Y = dataset.values[:,4].astype(str)

# Training data set
X_train = dataset.values[:125, 0:4]
# Corresponding training labels
y_train = dataset.values[:125, 4]

# Test data set
X_test = dataset.values[125:, 0:4]
# Corresponding test labels
y_test = dataset.values[125:, 4]

# Decision tree learning
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

# Predictions
print("Predicting labels")
results = dt.predict(X_test)

# Are the predicted results correct? Calculate accuracy by comparing results to the actual y_test labels
correct = 0
wrong = 0
for i in range(len(y_test)):
    if (y_test[i]==results[i]):
        correct+=1
    else:
        wrong+=1

print(f"Correct Predictions {correct} out of {len(y_test)} (Accuracy: {(correct/len(y_test))*100:.2f}%)")


        