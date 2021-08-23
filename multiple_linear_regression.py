# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 02:37:51 2021

@author: Gamerz
"""

#Multiple Linear Regression
# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values

# : means all the lines and :-1 all the column except the last 1 of the dataset

Y=dataset.iloc[:, 4].values
# Encoding categorical data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X= np.array(columnTransformer.fit_transform(X), dtype= np.str)

# avoiding dummy trap
X=X[:, 1:]



# making corelation between training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)"""

#fitting multiple linear regression

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set results

Y_pred= regressor.predict(X_test)

#building the optimal model with backward elimination
import statsmodels.api as sm
X= np.append(arr= np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt= X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS= sm.OLS(endog =Y, exog=X_opt).fit()