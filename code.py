# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:58:28 2018

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics

allowedPercentageOfMissingData = 50

fileName = "train.csv"
rawData = pd.read_csv(fileName, delimiter = ",")

#print(rawData.shape)
#print(rawData.head())
#print(rawData.info())

missingValues = rawData.isnull().sum()
missingValuesPercentage = 100 * rawData.isnull().sum() / len(rawData)
        
missingValuesTable = pd.concat([missingValues,missingValuesPercentage], axis=1)
missingValuesTable = missingValuesTable.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
missingValuesTable = missingValuesTable[missingValuesTable.iloc[:,1] != 0].sort_values('% of Total Values', ascending = False).round(1)

#print(missingValuesTable)

listOfMissingColumns = list(missingValuesTable[missingValuesTable['% of Total Values'] > allowedPercentageOfMissingData].index)
rawData = rawData.drop(list(listOfMissingColumns), axis = 1)

for column in rawData.columns:
    if rawData[column].dtype.kind in 'if':
        rawData[column].fillna(rawData[column].median(), inplace = True)
    else:
        rawData[column].fillna(rawData[column].value_counts().idxmax(), inplace = True) 

target = rawData['SalePrice']
rawData.drop(['SalePrice'], axis=1, inplace = True)

rawData = pd.get_dummies(rawData)

XTrain, XTest, yTrain, yTest = train_test_split(rawData, target, test_size = 0.2)

#function stealen from https://www.kaggle.com/marknagelberg/rmsle-function
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))

print('-------LineraRegression-------')
LR = LinearRegression()
LR.fit(XTrain,yTrain)
predictedValues = LR.predict(XTest)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(predictedValues, yTest)))
print('Root Mean Squared Log Error:', rmsle(predictedValues, yTest))


print('-------LassoRegression-------')
lasso = Lasso(alpha = 0.01, max_iter = 10e5)
lasso.fit(XTrain,yTrain)
predictedValues = lasso.predict(XTest)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(predictedValues, yTest)))
print('Root Mean Squared Log Error:', rmsle(predictedValues, yTest))


print('-------XGBoost--------')
model = xgb.XGBRegressor(n_estimators = 100, learning_rate = 0.05, gamma = 0.1, subsample = 0.75,colsample_bytree = 1, max_depth=10)

model.fit(XTrain, yTrain)
predictedValues = model.predict(XTest)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(predictedValues, yTest)))
print('Root Mean Squared Log Error:', rmsle(predictedValues, yTest))

#plt.scatter(yTest, predictions)