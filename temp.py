# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os


a1 = pd.read_excel("D:\\Credit-Risk-DEtection\\case_study1.xlsx")
a2 = pd.read_excel("D:\\Credit-Risk-DEtection\\case_study2.xlsx")


df1 = a1.copy()
df2 = a2.copy()



 #removing null values
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed .append(i)
        
        
df2=df2.drop(columns_to_be_removed,axis=1)

for i in df2.columns:
    df2 = df2.loc[ df2[i] != -99999 ]

df1.isnull().sum()
df2.isnull().sum()

#checking for common column
for i in list(df1.columns):
    if i in list(df2.columns):
        print (i)
        
# Merge the two dataframes, inner join so that no nulls are present
df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )

#Now we will categorise our dataset on categorical and numerical and work on them differently.

# check how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
   


# Chi-square test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)
    
# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)
        
# VIF sequentially check

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0



for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)
    
    
    if vif_value <= 6:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)

   