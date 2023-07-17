# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 07:55:05 2021

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
from sklearn.metrics import RocCurveDisplay #plot_roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib
from plot_roc_curve_woPred_drawfigure import *
from pathlib import Path

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

##################################################0. Figure Setting
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
##################################################1. Prepartion for Dataset
## Load the data
## Load the data
Data_Sero_RiskScore6 = pandas.read_csv('SeroScore_FcR_Lasso/SeroScore.csv',index_col=0)
Data_Group = pandas.read_csv('Progressor_allmonths/Data_group.csv',index_col=0)

#Run each features
colnames = Data_Sero_RiskScore6.columns

feature_index = 290872-1
sel_features = colnames[feature_index]
Data_Sero = Data_Sero_RiskScore6[sel_features]

Data_Sero.to_csv(Path('Progressor_allmonths/Selected_SeroScore.csv') , index=False)