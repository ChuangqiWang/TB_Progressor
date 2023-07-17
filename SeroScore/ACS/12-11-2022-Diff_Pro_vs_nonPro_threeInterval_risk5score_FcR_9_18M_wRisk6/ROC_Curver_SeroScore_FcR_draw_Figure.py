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
from sklearn.metrics import  auc, RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib
from plot_roc_curve_woPred_drawfigure import *
import scipy.stats as st
##################################################0. Figure Setting
font = {'family' : 'Arial',
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
Data_Sero_RiskScore6 = pandas.read_csv('SeroScore_FcR_Lasso/SeroScore.csv',index_col=0)
Data_Group = pandas.read_csv('../12-14-2022-Diff_Pro_vs_nonPro_9_18M_risk5score_FcR_woFunctional/Progressor_9_18months/Data_group.csv',index_col=0)


K_folder = 5
################################################## Create the folder
root = "SeroScore_ROC_Lasso"
if not os.path.exists(root):
        os.makedirs(root)

##################################################2. Many Iterations


#class_weight={0: 1, 1: w}
#model = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)

#Run each features
colnames = Data_Sero_RiskScore6.columns
for feature_index in [0]: #range(Data_Sero_RiskScore6.shape[1]):
    sel_features = colnames[feature_index]
    sel_features_saved = sel_features.replace('/', '_')
    ################################################## Create the folder
    directory = root + "/" + sel_features_saved
    if not os.path.exists(directory):
        os.makedirs(directory)
    #Run 100 iterations
    plt.rcParams["figure.figsize"] = (3, 3)
    fig, ax = plt.subplots()
    #Saving the parameter
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 50)
    total_Acc = []  
    for iteration in range(50):  
        Data_Sero = Data_Sero_RiskScore6[sel_features]
        #Dataframe to numpy
        X = Data_Sero.to_numpy()
        y = Data_Group['x']
        y = y == 'progressor'
        y = y.array
    
        
        #print("Iteration: %d" %(iteration))
        #Get the training and test set
        train_index = pandas.read_csv('../12-14-2022-Diff_Pro_vs_nonPro_9_18M_risk5score_FcR_woFunctional/Classification_RF_LASSO/idx_train_' + str(iteration+1) + '.csv')
        train_index = train_index["Resample1"] - 1 #R and Python transformation
        test_index = np.delete(np.arange(len(y)), train_index)
        #Train
        X_train = X[train_index, ]
        y_train = y[train_index]
        #Test
        X_test = X[test_index, ]
        y_test = y[test_index]
        #X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=iteration, stratify = y)
        yhat = X_train

        np.savez(os.path.join(directory, 'Iteration_' + str(iteration) + '_pred_actual_test.npz'), X_test, yhat, y_test)
        
        
        ###### 4. ROC plot
        viz = plot_roc_curve_woPred_drawfigure(yhat, y_train, name='Iteration {}'.format(iteration), alpha=0.3, lw=0.4, ax=ax)
        #viz = plot_roc_curve_woPred(yhat, y_train, name='Iteration {}'.format(iteration), alpha=0.3, lw=1)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
            
    np.savez(os.path.join(directory, 'AUCparameter' + '.npz'), tprs, aucs, viz)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='dimgray',  label='Chance', alpha=.8)
    #mean AUC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #low_auc, upper_auc = st.norm.interval(confidence=0.95, loc=np.mean(np.array(aucs)), scale=st.sem(np.array(aucs))) 
    low_auc, upper_auc = st.norm.interval(confidence=0.95, loc=np.mean(np.array(aucs)), scale=std_auc) 
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'AUC = %0.2f (%0.2f, %0.2f)' % (mean_auc, low_auc, upper_auc),
            lw=0.5, alpha=.8)
    
    # ax.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #         lw=2, alpha=.8)
    np.savez(os.path.join(root, 'AUCparameter_mean_' + sel_features_saved + '.npz'), fpr = mean_fpr, auc = mean_auc, tpr = mean_tpr)
    #std AUC
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    #Plot
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title="ROC:" + sel_features)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[index] for index in range(51, 52)], [labels[index] for index in range(51, 52)], loc="lower right", fontsize = "8")
    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.savefig(os.path.join(directory, sel_features_saved + '.Sero.Data.LassoFeatures.HyperParameters.png'))
    #plt.savefig(os.path.join(root, sel_features_saved + '.Sero.Data.LassoFeatures.HyperParameters.png'))
    plt.savefig(os.path.join(directory, sel_features_saved + '.Sero.Data.LassoFeatures.HyperParameters_Moresamples_V4.pdf'), format="pdf", bbox_inches="tight")
   
    plt.show()