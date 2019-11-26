
#IMPORT ALL USEFUL LIBRARIES
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#READ THE FILES REQUIRED FOR CLEANING 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


print(train.info())
#FILL MISSING VALUES OF TRAIN SET
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#FILL MISSING VALUES OF TEST DATA
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#REMOVE OUTLIERS FROM TRAIN AND TEST DATA
train['LoanAmount_log'] = np.log(train['LoanAmount'])#remove rigth skewness 
test['LoanAmount_log'] = np.log(test['LoanAmount'])

train.to_csv('clean_train.csv')
test.to_csv('clean_test.csv')

