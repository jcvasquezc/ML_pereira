# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:05:24 2017

@author: Andy VE Swimmer 16
"""


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot  as plt
import seasonal

# Reading data
TD = pd.read_excel("C:/Andres/NeuraLab/MLCourse/Regression/data/TD.xlsx")

# Casting to a Pandas DataFrame
TD = pd.DataFrame(TD)

# Extracting TD colum
TDC = TD["TD"]
TDT = TD["Fecha"]


# Visualizing TDC

plt.plot(TD["Index"],TD["TD"],marker='o')
plt.ylabel('Unemployment Percentage')
plt.show()


###########################################

TDnp = np.matrix(TD)
[X , Y]= [TDnp[:,0],TDnp[:,3]]
nn = MLPRegressor(activation="logistic",hidden_layer_sizes=500,validation_fraction = 0.8)
nn.fit(X,Y)

predictedValues = nn.predict(TDnp[:,0])

###########################################

plt.plot(TD["Index"],TD["TD"],marker='o')
plt.plot(TD["Index"],predictedValues,marker='o')
plt.ylabel('Unemployment Percentage')
plt.show()

###########################################

SSF = seasonal.fit_trend(TD["TD"],periodogram_thresh=0.1)
SSFvalues = SSF[1]


plt.plot(TD["Index"],TD["TD"],marker='o')
plt.plot(TD["Index"],predictedValues)
plt.plot(TD["Index"],SSFvalues)
plt.ylabel('Unemployment Percentage')
plt.show()
