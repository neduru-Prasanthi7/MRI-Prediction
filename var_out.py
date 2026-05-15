import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('var_outl')
from scipy import stats

def variable_transformation_outliers(X_train,X_test):
    try:
        #logger.info(f' Before Total rows in training data : {X_train.shape}')
        #logger.info(f'Total rows in testing data  : {X_test.shape}')
        #for i in X_train.columns:
        #    X_train[i].plot(kind= 'kde',color='blue')
        #    plt.show()
        #for i in X_train.columns:
        #    sns.boxplot(x = X_train[i])
        #    plt.show()
        logger.info(f'Before training data {X_train.columns}--> {X_train.shape}')
        logger.info(f'Before testing data :{X_test.columns} --> {X_test.shape}')
        for i in X_train.columns:
            X_train[i+'_yeo'],lam =stats.yeojohnson(X_train[i])
            X_train = X_train.drop([i],axis = 1)
            iqr = X_train[i+'_yeo'].quantile(0.75) - X_train[i+'_yeo'].quantile(0.25)
            upper_limit = X_train[i+'_yeo'].quantile(0.75) + (1.5 * iqr)
            lower_limit = X_train[i+'_yeo'].quantile(0.25) - (1.5 * iqr)
            X_train[i+'_yeo_trim'] = np.where(X_train[i+'_yeo'] > upper_limit,upper_limit,
                                              np.where(X_train[i+'_yeo'] < lower_limit,lower_limit,X_train[i+'_yeo']))
            X_train = X_train.drop([i+'_yeo'],axis = 1)

            X_test[i + '_yeo_trim'] = np.where(X_test[i] > upper_limit, upper_limit,
                                                np.where(X_test[i] < lower_limit, lower_limit,
                                                         X_test[i]))
            X_test = X_test.drop([i], axis=1)

        logger.info(f'After training data :{X_train.columns} ---->{X_train.shape}')
        logger.info(f'After testing data :{X_test.columns} ---->{X_test.shape}')

        #for i in X_train.columns:
         #   sns.boxplot(x = X_train[i])
         #   plt.show()


        return X_train,X_test


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')