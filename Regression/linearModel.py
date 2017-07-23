# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:04:00 2017

@author: Andy VE Swimmer 16
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot  as plt

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
lm = linear_model.LinearRegression()
lm.fit(X,Y)

original_score = '{0:.3f}'.format( lm.score( X, Y ) )

predictedValues = lm.predict(TDnp[:,0])

###########################################

plt.plot(TD["Index"],TD["TD"],marker='o')
plt.plot(TD["Index"],predictedValues,marker='o')
plt.ylabel('Unemployment Percentage')
plt.show()

###########################################
