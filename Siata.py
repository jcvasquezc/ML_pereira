# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 20:33:47 2017

@author: Andy VE Swimmer 16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn import linear_model
import math
##################

def toStr(x):
    x = str(x) + ".csv"
    return(x)

##################
# import data

FileNames = list(range(2012,2018,1))

FileNames = map(toStr,FileNames)
FileNames = list(FileNames)

dataSiata = pd.read_csv(FileNames[0])

for i in FileNames[1:len(FileNames)]:
    dataSiata = dataSiata.append(pd.read_csv(i))

sequence = list(range(0,len(dataSiata)))

##################
# plot Data
plt.plot_date(dataSiata["fecha"],dataSiata["pm25"],fmt="r-",color = "blue")
plt.plot_date(dataSiata["fecha"],dataSiata["pm10"],fmt="r-",color = "red")
plt.plot_date(dataSiata["fecha"],dataSiata["ozono"],fmt="r-",color = "green")
plt.plot_date(dataSiata["fecha"],dataSiata["nox"],fmt="r-",color = "green")
plt.title("Air Quality")
plt.grid(True)

##################
# correlation

dataSiata.corr("pearson")

dataSiata.corr("spearman")


##################
# plot Data
plt.plot_date(dataSiata["fecha"],dataSiata["pm25"],fmt="r-",color = "blue")
plt.plot_date(dataSiata["fecha"],dataSiata["pm10"],fmt="r-",color = "red")
plt.title("Air Quality")
plt.grid(True)

##################
# filter data
dataSiataNOX = dataSiata[dataSiata["nox"] != 0]


##################
# plot Data
plt.plot_date(dataSiataNOX["fecha"],dataSiataNOX["pm25"],fmt="r-",color = "blue")
plt.plot_date(dataSiataNOX["fecha"],dataSiataNOX["pm10"],fmt="r-",color = "red")
plt.plot_date(dataSiataNOX["fecha"],dataSiataNOX["ozono"],fmt="r-",color = "green")
plt.plot_date(dataSiataNOX["fecha"],dataSiataNOX["nox"],fmt="r-",color = "green")
plt.title("Air Quality")
plt.grid(True)


##################
# plot Data
plt.scatter(dataSiataNOX["pm25"],dataSiataNOX["nox"],color = "blue")
plt.scatter(dataSiataNOX["pm10"],dataSiataNOX["nox"],color = "red")
plt.scatter(dataSiataNOX["ozono"],dataSiataNOX["nox"],color = "green")
##################
# split data

# Train
train = dataSiataNOX.iloc[0:5860]

p10Part = train["pm10"]
nox = train["nox"]

p10PartNP = np.matrix(p10Part)
noxNP = np.matrix(nox)


# Test
test = dataSiataNOX.iloc[5860:5882]

p10PartTest = test["pm10"]
noxTest = test["nox"]

p10PartNPTest = np.matrix(p10PartTest)
noxNPTest = np.matrix(noxTest)

##################
# linear model

lm = linear_model.LinearRegression()
fittedlm = lm.fit(p10PartNP.transpose(),noxNP.transpose())

fittedlm.coef_
fittedlm.intercept_

##################
# predictions linear model train
noxFitted = fittedlm.predict(p10PartNP.transpose())
noxFitted = pd.DataFrame(noxFitted,columns = ["predictionNOX"])

plt.scatter(p10Part,nox,color = "red")
plt.scatter(p10Part,noxFitted["predictionNOX"],color = "green")

#
plt.plot_date(train["fecha"],train["nox"],fmt="r-",color = "red")
plt.plot_date(train["fecha"],noxFitted["predictionNOX"],fmt="r-",color = "green")


##################
# predictions linear model test
noxFittedTest = fittedlm.predict(p10PartNPTest.transpose())
noxFittedTest = pd.DataFrame(noxFittedTest,columns = ["predictionNOX"])

plt.scatter(p10PartTest,noxTest,color = "red")
plt.scatter(p10PartTest,noxFittedTest["predictionNOX"],color = "green")

#
plt.plot_date(test["fecha"],test["nox"],fmt="r-",color = "red")
plt.plot_date(test["fecha"],noxFittedTest["predictionNOX"],fmt="r-",color = "green")


###################################
# Log transformation train data
p10PartLog = p10Part.apply(math.log)
noxLog = nox.apply(math.log)

p10PartNPLog = np.matrix(p10PartLog)
noxNPLog = np.matrix(noxLog)

###################################
# Log transformation test data
p10PartTestLog = p10PartTest.apply(math.log)
noxTestLog = noxTest.apply(math.log)

p10PartNPTestLog = np.matrix(p10PartTestLog)
noxNPTestLog = np.matrix(noxTestLog)


##################
# linear model transformed data

fittedlmLog = lm.fit(p10PartNPLog.transpose(),noxNPLog.transpose())


##################
# predictions linear model train transformed data
noxFittedLog = fittedlmLog.predict(p10PartNPLog.transpose())
noxFittedLog = pd.DataFrame(noxFittedLog,columns = ["predictionNOX"])

plt.scatter(p10PartLog,noxLog,color = "red")
plt.scatter(p10PartLog,noxFittedLog["predictionNOX"],color = "green")

#
plt.plot_date(train["fecha"],noxLog,fmt="r-",color = "green")
plt.plot_date(train["fecha"],noxFittedLog["predictionNOX"],"r-",color = "red")


# predictions linear model train transformed data
noxFittedTestLog = fittedlmLog.predict(p10PartNPTestLog.transpose())
noxFittedTestLog = pd.DataFrame(noxFittedTestLog,columns = ["predictionNOX"])

plt.scatter(p10PartTestLog,noxTestLog,color = "red")
plt.scatter(p10PartTestLog,noxFittedTestLog["predictionNOX"],color = "green")

#
plt.plot_date(test["fecha"],p10PartTestLog,fmt="r-",color = "green")
plt.plot_date(test["fecha"],noxFittedTestLog["predictionNOX"],"r-",color = "red")


plt.plot_date(test["fecha"],p10PartTestLog,fmt="r-",color = "green")
plt.plot_date(test["fecha"],noxFittedTestLog["predictionNOX"],"r-",color = "red")

plt.plot_date(test["fecha"],p10PartTestLog.apply(math.exp),fmt="r-",color = "green")
plt.plot_date(test["fecha"],noxFittedTestLog["predictionNOX"].apply(math.exp),"r-",color = "red")






