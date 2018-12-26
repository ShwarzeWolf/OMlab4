# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:58:28 2018

@author: DreamTeam
"""
#import libraries to work
import pandas as pd
from sklearn.model_selection import train_test_split

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

X_train, X_test, y_train, y_test = train_test_split(rawData, target, test_size=0.2,random_state=2)

