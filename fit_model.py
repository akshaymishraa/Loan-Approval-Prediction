# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 22:47:49 2019

@author: Akshay
"""
import pandas as pd
train=pd.read_csv('clean_train.csv')
test=pd.read_csv('clean_test.csv')

train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

X = train.drop('Loan_Status', 1)
y = train.Loan_Status

#add code for test input and append to tes data

X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)




from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
print(accuracy_score(y_cv, pred_cv))
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_cv, pred_cv)
print(cm)
pred_test = model.predict(test)
print(pred_test)