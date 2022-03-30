# -*- Coding = UTF-8 -*-
# @Time: 2022/3/30 18:10
# @Author: Nico
# File: LogisticRegression-Matrix.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Demo\MachineLearning\LogisticRegression\dataset.csv')
dataset = dataset[['Age', 'Pclass', 'Sex', 'Fare', 'SibSp', 'Parch', 'Survived']]
n_dataset = dataset.shape[0]

# feature engineering
dataset['Fare'] = (dataset['Fare'] - dataset['Fare'].mean()) / dataset['Fare'].std()
dataset['Sex'] = dataset['Sex'].astype('category').cat.codes
dataset['Pclass'] = (dataset['Pclass'] - dataset['Pclass'].mean()) / dataset['Pclass'].std()
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset['Age'] = (dataset['Age'] - dataset['Age'].mean()) / dataset['Age'].std()
dataset['Sex'] = (dataset['Sex'] - dataset['Sex'].mean()) / dataset['Sex'].std()

train_data = dataset.iloc[0:600]
test_data = dataset.iloc[600:]

# print(dataset.head())

X_train = train_data.drop(columns='Survived').astype('float32')
y_train = train_data['Survived'].astype('float32')
n_train = train_data.shape[0]

X_test = test_data.drop(columns='Survived').astype('float32')
y_test = test_data['Survived'].astype('float32')
n_test = test_data.shape[0]

# print(X_train.head())
# print(y_train.head())


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


x = np.arange(-10, 10, 0.1)
y = sigmoid(x)
fig, ax = plt.subplots()
ax.scatter(0, sigmoid(0))
ax.plot(x, y)
plt.show()

# Build Logistic Regression Model
n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0

N = 1000
lr = 0.0001

for j in range(N):
    det_w = np.zeros(n_features)
    det_b = 0.0

    logits = w.dot(X_train.T) + b
    y_hat = sigmoid(logits)

    det_w = - np.dot((y_train - y_hat), X_train)
    det_b = - np.sum(y_train - y_hat)

    w = w - lr * det_w
    b = b - lr * det_b


def get_accuracy(X, y, w, n):
    n_samples = X.shape[0]
    predict_result = []
    for i in range(n_samples):
        x = X.iloc[i]
        p = sigmoid(x.dot(w) + b)
        if p > 0.5:
            predict_result.append(1)
        else:
            predict_result.append(0)
    total_loss = 0
    for i in range(n_samples):
        loss = (y.iloc[i] - predict_result[i]) ** 2
        total_loss += loss
    accuracy = (y.shape[0] - total_loss) / y.shape[0]
    return accuracy


print(get_accuracy(X_train, y_train, w, b))
print(get_accuracy(X_test, y_test, w, b))

