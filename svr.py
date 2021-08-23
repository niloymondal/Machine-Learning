# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 01:51:04 2021

@author: Gamerz
"""

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X =sc_X.fit_transform(X)
Y =sc_Y.fit_transform(Y)

#fitting SVR in the dataset
from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(np.reshape(X, (-1, 1), Y))



Y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#visualizing the SVR regression results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()