# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 19:30:04 2021

@author: Gamerz
"""

#polynomial regression
# Importing the Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:, 1:2].values

# : means all the lines and :-1 all the column except the last 1 of the dataset

Y=dataset.iloc[:, 2].values



# making corelation between training set and test set

"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)"""

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)"""

#fitting linear regression in the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
#fitting polynomial regression in the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2= LinearRegression()
lin_reg_2.fit(X_poly, Y)

#visualizing the linear regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#visualizing the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with the linear regression
lin_reg.predict([[6.5]])
#predicting a new result with the polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))