# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:58:28 2018

@author: DreamTeam
"""
#import libraries to work
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import RobustScaler
import warnings
import xgboost as xgb
from hyperopt import hp
from h import fmin, tpe, STATUS_OK, Trials

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

def rmsle(ypred, ytest): 
    assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))

def print_error_info(y_test,predicted):
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, predicted)))
    print("RMSLE: ", rmsle(y_test, predicted))

fileName = "train.csv"

rawData = pd.read_csv(fileName, delimiter = ",")

#print(rawData.shape)
#print(rawData.head())
#print(rawData.info())

# Function to calculate missing values by column
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
       
        return mis_val_table_ren_columns
    
# Get the columns with > 50% missing
missing_df = missing_values_table(rawData);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)

# Drop the columns
rawData = rawData.drop(list(missing_columns), axis = 1)

for column in rawData.columns:
    if rawData[column].dtype.kind in 'if':
        rawData[column].fillna(rawData[column].median(), inplace=True)
    else:
        rawData[column].fillna(rawData[column].value_counts().idxmax(),inplace=True) 

target = rawData['SalePrice']
rawData.drop(['SalePrice'],axis=1,inplace=True)

rawData = pd.get_dummies(rawData)

X_train, X_test, y_train, y_test = train_test_split(rawData, target, test_size=0.2)


#LineraRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
predicted = LR.predict(X_test)

print_error_info(predicted,y_test)

print(LR.coef_.shape)
print(LR.coef_)

plt.scatter(y_test, predicted)

#LassoRegression
lasso = LassoCV(alphas=np.logspace(-1,4,25))

lasso.fit(X_train,y_train)

predicted = lasso.predict(X_test)
print_error_info(predicted, y_test)

print(lasso.coef_.shape)

plt.scatter(y_test, predicted)

#XGBoost

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

traindf, testdf = train_test_split(X_train, test_size = 0.3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print_error_info(y_test, predictions)

plt.scatter(y_test, predictions)