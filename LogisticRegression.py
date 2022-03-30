# -*- Coding = UTF-8 -*-
# @Time: 2022/3/29 22:27
# @Author: Nico
# File: LogisticRegression.py
# @Software: PyCharm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Demo\MachineLearning\LogisticRegression\dataset.csv')
dataset = dataset[['Pclass', 'Sex', 'Fare', 'Survived']]
n_dataset = dataset.shape[0]

# feature engineering
dataset['Fare'] = (dataset['Fare'] - dataset['Fare'].mean()) / dataset['Fare'].std()
dataset['Sex'] = dataset['Sex'].astype('category').cat.codes

train_data = dataset.iloc[0:600]
test_data = dataset.iloc[600:]

# print(dataset.head())

X_train = train_data.drop(columns='Survived').astype('float32')
y_train = train_data['Survived'].astype('float32')
n_train = train_data.shape[0]

X_test = test_data.drop(columns='Survived').astype('float32')
y_test = test_data['Survived'].astype('float32')
n_test = test_data.shape[0]


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
w1 = -0.5
w2 = -2
w3 = 0.3
b = 2

N = 1000
lr = 0.0001

for j in range(N):
    det_w1 = 0
    det_w2 = 0
    det_w3 = 0
    det_b = 0
    for i in range(n_train):
        x1 = X_train.iloc[i][0]
        x2 = X_train.iloc[i][1]
        x3 = X_train.iloc[i][2]
        z = w1 * x1 + w2 * x2 + w3 * x3 + b
        y_hat = sigmoid(z)
        y = y_train.iloc[i]

        det_w1 += - (y - y_hat) * x1
        det_w2 += - (y - y_hat) * x2
        det_w3 += - (y - y_hat) * x3
        det_b += - (y - y_hat)

    w1 = w1 - lr * det_w1
    w2 = w2 - lr * det_w2
    w3 = w3 - lr * det_w3
    b = b - lr * det_b


def get_accuracy(X, y, n):
    predicted_result = []
    total_loss = 0
    for i in range(n):
        x1 = X.iloc[i][0]
        x2 = X.iloc[i][1]
        x3 = X.iloc[i][2]
        z = w1 * x1 + w2 * x2 + w3 * x3 + b
        y_hat = sigmoid(z)
        if y_hat < 0.5:
            predicted_result.append(0)
        else:
            predicted_result.append(1)
    for i in range(n):
        loss = (y.iloc[i] - predicted_result[i]) ** 2
        total_loss += loss
    accuracy = (n - total_loss) / n
    return accuracy


print(get_accuracy(X_train, y_train, n_train))
print(get_accuracy(X_test, y_test, n_test))

