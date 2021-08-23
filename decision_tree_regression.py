# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 22:16:11 2021

@author: Gamerz
"""

#Decision tree regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:, 1:2].values
X =X.reshape(-1,1)
# : means all the lines and :-1 all the column except the last 1 of the dataset

Y=dataset.iloc[:, 2].values
Y = Y.reshape (-1,1)


# making corelation between training set and test set

"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)"""

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)"""

#fitting Decision Tree Regression 
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)
#predicting the test results
Y_pred= regressor.predict([[6.5]]).reshape(1,1)

#visualizing the Decision Tree Regression 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
