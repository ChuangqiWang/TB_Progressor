# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 07:16:13 2021

@author: Chuangqi
"""

#https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/

import pandas
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn import datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import plot_roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib
from plot_roc_curve_woPred import *
import pandas as pd

## Load the data
Data_Sero_RiskScore6 = pandas.read_csv('Progressor_18_27months/Data_combining.csv',index_col=0)
Data_Group = pandas.read_csv('Progressor_18_27months/Data_group.csv',index_col=0)

################################################## Create the folder
directory = "PerFeatures_ROC_v2"
if not os.path.exists(directory):
        os.makedirs(directory)
        
        
#Run each features
colnames = Data_Sero_RiskScore6.columns
df_AUC = pd.DataFrame(np.zeros((1, len(colnames))),
                   columns = colnames)
for feature_index in range(Data_Sero_RiskScore6.shape[1]):
    sel_features = colnames[feature_index]
    sel_features_saved = sel_features.replace('/', '_')
    ################################################## Create the folder
    directory = "PerFeatures_ROC_v2" + "/" + sel_features_saved
    if not os.path.exists(directory):
        os.makedirs(directory) 
    saved_results = np.load(os.path.join(directory, 'AUCparameter_mean' + '.npz'))
    mean_auc = saved_results['auc']
    df_AUC.at[0, sel_features] = mean_auc
    
    df_AUC.to_csv('mean_AUC.csv', index=False)
    